"""Synthetic inference experiment: generate -> fit -> forecast -> private evaluation.

No renderer, engine, camera, GPU, model weights or external assets required.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest, write_json
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture, DriveState, cycle, nominal_remote_peak
from wmfracture.compliant_inference import (PrefixModel, equilibrium_cycle, damage, grid,
    log_likelihood, normalize, ParticleFilter, weighted_quantile)
from wmfracture.joint_state import matched_history


def make_cases(config, law, fixture):
    known = {'law':asdict(law)}; del known['law']['fracture_energy']
    cases = []
    for ratio in config['matched_stiffness_ratios']:
        for energy in config['matched_energies']:
            material = replace(law, fracture_energy=energy)
            intended = matched_history(known, energy, ratio)
            command = nominal_remote_peak(material, fixture, intended)
            cases.append({'parameters':[energy, command], 'family':'matched_stiffness', 'in_prior':True})
    rng = np.random.default_rng(config['seed'])
    bounds = np.asarray(config['parameter_bounds'])
    for unit in rng.uniform(.05, .95, (config['random_in_prior_cases'], 2)):
        cases.append({'parameters':(bounds[:, 0]+unit*np.ptp(bounds, axis=1)).tolist(),
            'family':'random', 'in_prior':True})
    for parameters in config['out_of_prior_parameters']:
        cases.append({'parameters':parameters, 'family':'out_of_prior', 'in_prior':False})
    for index, case in enumerate(cases):
        case['id'] = f'case-{index:03d}'
        unit = (np.asarray(case['parameters'])-bounds[:, 0])/np.ptp(bounds, axis=1)
        assert bool(((unit >= 0) & (unit <= 1)).all()) == case['in_prior']
    return cases


def truth_prefix(parameters, actions, law, fixture):
    energy, prep = parameters
    _, maximum = equilibrium_cycle(0., energy, prep, law, fixture)
    values, maxima = [], []
    for action in actions:
        gap, maximum = equilibrium_cycle(maximum, energy, action, law, fixture)
        values.append(float(gap)); maxima.append(float(maximum))
    return values, maxima


def generate(root, config, law, fixture):
    cases = make_cases(config, law, fixture)
    write_json(root/'private-cases.json', cases)
    episodes = []
    for case_index, case in enumerate(cases):
        for sigma_index, sigma in enumerate(config['gap_noise_sigma_m']):
            # Paired noise across protocols/scenarios; this is not independent replication.
            rng = np.random.default_rng(config['seed']+10000+100*case_index+sigma_index)
            noise = rng.normal(0, sigma, 3)
            for protocol, actions in config['protocols'].items():
                exact, _ = truth_prefix(case['parameters'], actions, law, fixture)
                for scenario, bias in config['observation_scenarios'].items():
                    episodes.append({'id':f'episode-{len(episodes):04d}', 'case_id':case['id'],
                        'protocol':protocol, 'scenario':scenario, 'sigma_m':sigma,
                        'actions_m':actions, 'observed_gaps_m':(np.asarray(exact)+noise+bias).tolist()})
    write_json(root/'observations.json', episodes)
    return cases


def dynamic_audit(config, cases, law, fixture):
    audit = config['dynamic_audit']; records = []
    # Continuous inertial solver, including physical preparation/unload/rest.
    actions = config['protocols']['fixed_probe']
    for index in audit['case_indices']:
        energy, prep = cases[index]['parameters']
        material = replace(law, fracture_energy=energy)
        for dt in audit['dt_levels']:
            state = DriveState()
            for stage, command in enumerate([prep]+actions):
                trace, _ = cycle(state, material, fixture, command, dt, **audit['cycle'])
                held = [r for r in trace if r['segment'] == 'hold'][-1]
                expected, maxima = truth_prefix([energy, prep], actions[:stage], law, fixture)
                _, prep_max = equilibrium_cycle(0., energy, prep, law, fixture)
                expected_gap = float(prep_max) if stage == 0 else expected[-1]
                expected_max = float(prep_max) if stage == 0 else maxima[-1]
                records.append({'case_id':cases[index]['id'], 'dt':dt, 'stage':stage,
                    'hold_gap_error_m':abs(held['gap']-expected_gap),
                    'post_cycle_maximum_error_m':abs(state.maximum_opening-expected_max),
                    'end_gap_m':state.gap, 'end_velocity_m_s':state.velocity})
            for command in config['future_drives_m']:
                from wmfracture.compliant_fixture import clone_state
                branch = clone_state(state)
                trace, _ = cycle(branch, material, fixture, command, dt, **audit['cycle'])
                held = [r for r in trace if r['segment'] == 'hold'][-1]
                expected, _ = equilibrium_cycle(state.maximum_opening, energy, command, law, fixture)
                records.append({'case_id':cases[index]['id'], 'dt':dt, 'future_drive_m':command,
                    'hold_gap_error_m':abs(held['gap']-float(expected)),
                    'failure_event_agrees':bool((held['damage'] == 1.) == (command >= material.failure_opening))})
        print(f'DYNAMIC_AUDIT {cases[index]["id"]}', flush=True)
    return {'records':records, 'maximum_hold_gap_error_m':max(r['hold_gap_error_m'] for r in records),
        'maximum_post_cycle_history_error_m':max(r.get('post_cycle_maximum_error_m', 0.) for r in records),
        'passed':all(r['hold_gap_error_m'] < audit['maximum_hold_gap_error_m'] and r.get('failure_event_agrees', True) for r in records),
        'scope':'Selected slow stable-branch cases only; no transient trajectory or unstable-branch equivalence claim.'}


def fit(root, config, law, fixture):
    # This stage reads noisy observations, never private-cases.json or futures.
    episodes = json.loads((root/'observations.json').read_text())
    model = PrefixModel(law, fixture, config['parameter_bounds'])
    archive, packets, bank = {}, [], {}
    for side in config['grid_sides']:
        unit = grid(side); archive[f'grid{side}'] = unit
        for protocol, actions in config['protocols'].items():
            prediction, maximum = model.predict(unit, actions)
            bank[(side, protocol)] = prediction
    library_cost = dict(model.cost)
    for episode_index, episode in enumerate(episodes):
        y, sigma = np.asarray(episode['observed_gaps_m']), episode['sigma_m']
        protocol = episode['protocol']
        for stage in range(1, len(y)+1):
            for side in config['grid_sides']:
                pred = bank[(side, protocol)][:, :stage]
                weights = normalize(log_likelihood(pred, y[:stage], sigma))
                methods = [(f'grid{side}_soft', weights)]
                if side == 7:
                    keep = np.all(np.abs(pred-y[:stage]) <= config['hard_sigma_multiple']*sigma, axis=1)
                    methods.append(('grid7_hard', keep/keep.sum() if keep.any() else np.zeros(len(keep))))
                for name, w in methods:
                    key = f'packet{len(packets)}'; archive[key+'_weights'] = w
                    packets.append({'episode_id':episode['id'], 'stage':stage, 'method':name,
                        'unit_key':f'grid{side}', 'weight_key':key+'_weights',
                        'cost':{'parameter_prefix_evaluations':side*side,
                            'equilibrium_cycle_evaluations':side*side*(1+stage)},
                        'cost_scope':'Counterfactual independent fit cost; reusable grid library separately recorded.'})
        for move_steps in (0, config['smc']['move_steps']):
            for replicate in range(config['smc']['replicates']):
                # Identical initial particles/RNG for paired baseline comparisons.
                seed = config['seed']+100000+episode_index*100+replicate
                model.reset_cost()
                smc = ParticleFilter(model, config['smc']['particles'], seed, move_steps,
                    config['smc']['move_scale'], config['smc']['global_probability'])
                started = time.perf_counter()
                for stage in range(1, len(y)+1):
                    trace = smc.update(episode['actions_m'][:stage], y[:stage], sigma)
                    key = f'packet{len(packets)}'
                    archive[key+'_unit'], archive[key+'_weights'] = smc.unit.copy(), smc.weights.copy()
                    packets.append({'episode_id':episode['id'], 'stage':stage,
                        'method':'smc128_move' if move_steps else 'smc128_resample', 'replicate':replicate,
                        'unit_key':key+'_unit', 'weight_key':key+'_weights', 'trace':trace,
                        'cost':dict(model.cost), 'elapsed_fit_seconds':time.perf_counter()-started})
        if episode_index % 60 == 0:
            print(f'FIT {episode_index+1}/{len(episodes)}', flush=True)
    np.savez_compressed(root/'posteriors.npz', **archive)
    write_json(root/'fit-packets.json', packets)
    write_json(root/'library-cost.json', library_cost)


def forecast(root, config, law, fixture):
    # No private labels: forecasts are sealed before private evaluation.
    episodes = {e['id']:e for e in json.loads((root/'observations.json').read_text())}
    packets = json.loads((root/'fit-packets.json').read_text())
    archive = np.load(root/'posteriors.npz', allow_pickle=False)
    model = PrefixModel(law, fixture, config['parameter_bounds'])
    rows = []
    for packet in packets:
        episode = episodes[packet['episode_id']]
        unit, weights = archive[packet['unit_key']], archive[packet['weight_key']]
        nonzero = weights > 0
        row = {**packet, 'empty':not bool(nonzero.any()), 'futures':[]}
        if nonzero.any():
            unit, weights = unit[nonzero], weights[nonzero]
            _, maximum = model.predict(unit, episode['actions_m'][:packet['stage']])
            energy = model.parameters(unit)[:, 0]
            row['ess'] = float(1/np.sum(weights**2))
            for command in config['future_drives_m']:
                gap, after = equilibrium_cycle(maximum, energy, command, law, fixture)
                row['futures'].append({'drive_m':command, 'mean_gap_m':float(weights@gap),
                    'q025_gap_m':None if packet['method'].endswith('_hard') else weighted_quantile(gap, weights, .025),
                    'q975_gap_m':None if packet['method'].endswith('_hard') else weighted_quantile(gap, weights, .975),
                    'support_min_gap_m':float(gap.min()), 'support_max_gap_m':float(gap.max()),
                    'separation_mass':None if packet['method'].endswith('_hard') else float(weights@(damage(after, energy, law) >= 1-1e-12)),
                    'maximum_additional_damage_in_retained_samples':float(np.max(damage(after, energy, law)-damage(maximum, energy, law)))})
        rows.append(row)
    archive.close()
    write_json(root/'forecasts.json', rows)
    return {'posteriors':digest(root/'posteriors.npz'), 'packets':digest(root/'fit-packets.json'),
        'forecasts':digest(root/'forecasts.json')}


def evaluate(root, config, law, fixture, seals):
    cases = {c['id']:c for c in json.loads((root/'private-cases.json').read_text())}
    episodes = {e['id']:e for e in json.loads((root/'observations.json').read_text())}
    forecasts = json.loads((root/'forecasts.json').read_text())
    rows, physical = [], []
    for episode in episodes.values():
        case = cases[episode['case_id']]; energy, prep = case['parameters']
        _, maxima = truth_prefix(case['parameters'], episode['actions_m'], law, fixture)
        _, initial = equilibrium_cycle(0., energy, prep, law, fixture)
        for stage, maximum in enumerate(maxima, 1):
            # Repeated weak is the paired control for irreversible intervention cost.
            _, passive = truth_prefix(case['parameters'], [config['protocols']['repeated_weak'][0]]*stage, law, fixture)
            physical.append({'episode_id':episode['id'], 'stage':stage, 'in_prior':case['in_prior'],
                'protocol':episode['protocol'], 'preparation_damage':float(damage(initial, energy, law)),
                'extra_damage_over_repeated_weak':float(damage(maximum, energy, law)-damage(passive[-1], energy, law))})
    for prediction in forecasts:
        episode = episodes[prediction['episode_id']]; case = cases[episode['case_id']]
        energy, _ = case['parameters']
        _, maxima = truth_prefix(case['parameters'], episode['actions_m'][:prediction['stage']], law, fixture)
        for future in prediction['futures'] or [{}]:
            row = {'episode_id':episode['id'], 'case_id':case['id'], 'family':case['family'],
                'in_prior':case['in_prior'], 'protocol':episode['protocol'], 'scenario':episode['scenario'],
                'sigma_m':episode['sigma_m'], 'stage':prediction['stage'], 'method':prediction['method'],
                'replicate':prediction.get('replicate'), 'empty':prediction['empty'], 'cost':prediction['cost']}
            if future:
                exact, after = equilibrium_cycle(maxima[-1], energy, future['drive_m'], law, fixture)
                separated = bool(damage(after, energy, law) >= 1-1e-12)
                row.update({'drive_m':future['drive_m'], 'true_gap_m':float(exact), 'true_separated':separated,
                    'squared_error_mm2':float(((future['mean_gap_m']-exact)*1000)**2),
                    'covered_95':None if prediction['method'].endswith('_hard') else bool(future['q025_gap_m']-1e-12 <= exact <= future['q975_gap_m']+1e-12),
                    'covered_support':bool(future['support_min_gap_m']-1e-12 <= exact <= future['support_max_gap_m']+1e-12),
                    'brier':None if prediction['method'].endswith('_hard') else float((future['separation_mass']-separated)**2)})
            rows.append(row)
    write_json(root/'evaluation-rows.json', rows)
    write_json(root/'intervention-costs.json', physical)
    groups = []
    for in_prior in (True, False):
        for scenario in config['observation_scenarios']:
            for sigma in config['gap_noise_sigma_m']:
                for protocol in config['protocols']:
                    for stage in (1, 2, 3):
                        for method in sorted({r['method'] for r in rows}):
                            selected = [r for r in rows if (r['in_prior'], r['scenario'], r['sigma_m'], r['protocol'], r['stage'], r['method']) == (in_prior, scenario, sigma, protocol, stage, method)]
                            available = [r for r in selected if not r['empty']]
                            brier = [r['brier'] for r in available if r['brier'] is not None]
                            coverage = [r['covered_95'] for r in available if r['covered_95'] is not None]
                            unique = {(r['episode_id'], r['replicate']):r for r in selected}
                            groups.append({'in_prior':in_prior, 'scenario':scenario, 'sigma_m':sigma,
                                'protocol':protocol, 'stage':stage, 'method':method, 'episodes_with_replicates':len(unique),
                                'empty_fraction':sum(r['empty'] for r in unique.values())/len(unique),
                                'gap_rmse_mm':float(np.sqrt(np.mean([r['squared_error_mm2'] for r in available]))) if available else None,
                                'empirical_95_coverage':float(np.mean(coverage)) if coverage else None,
                                'empirical_support_coverage':float(np.mean([r['covered_support'] for r in available])) if available else None,
                                'separation_brier':float(np.mean(brier)) if brier else None,
                                'prefix_evaluations_per_fit':float(np.mean([r['cost']['parameter_prefix_evaluations'] for r in unique.values()]))})
    # Grid refinement checks the numerical approximation to the assumed posterior.
    lookup = {(p['episode_id'],p['stage'],p['method']):p for p in forecasts if p['method'] in ('grid41_soft','grid81_soft')}
    differences = []
    for (episode_id, stage, method), pred in lookup.items():
        if method != 'grid41_soft':
            continue
        fine = lookup[(episode_id, stage, 'grid81_soft')]
        for a,b in zip(pred['futures'], fine['futures']):
            differences.append({'episode_id':episode_id, 'stage':stage,
                'mean_gap_difference_mm':abs(a['mean_gap_m']-b['mean_gap_m'])*1000,
                'separation_mass_difference':abs(a['separation_mass']-b['separation_mass'])})
    write_json(root/'grid-refinement.json', differences)
    assert seals == {'posteriors':digest(root/'posteriors.npz'), 'packets':digest(root/'fit-packets.json'), 'forecasts':digest(root/'forecasts.json')}
    return groups, differences, physical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-dynamic-audit', action='store_true', help='Smoke runs only; audited status will be false')
    args = parser.parse_args()
    project = Path(os.environ['WM_PROJECT_ROOT'])
    config = json.loads((project/'config/synthetic-filter-benchmark.json').read_text())
    root = Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-synthetic-filter-benchmark')
    root.mkdir(parents=True); print(f'OUTPUT={root}', flush=True)
    write_json(root/'config.json', config)
    write_json(root/'code-sha256.json', {str(p.relative_to(project)):digest(p) for p in
        [Path(__file__), project/'wmfracture/compliant_inference.py', project/'wmfracture/compliant_fixture.py', project/'wmfracture/cohesive.py']})
    law = InterfaceLaw(**json.loads((project/config['law_config']).read_text())['law'])
    fixture = Fixture(**config['fixture']); started = time.perf_counter(); timings = {}
    cases = generate(root, config, law, fixture)
    audit = {'passed':False, 'skipped':True} if args.skip_dynamic_audit else dynamic_audit(config, cases, law, fixture)
    write_json(root/'dynamic-audit.json', audit); timings['generation_and_audit_seconds'] = time.perf_counter()-started
    tick = time.perf_counter(); fit(root, config, law, fixture); timings['fit_seconds'] = time.perf_counter()-tick
    tick = time.perf_counter(); seals = forecast(root, config, law, fixture); timings['forecast_seconds'] = time.perf_counter()-tick
    write_json(root/'sealed-predictions.json', seals)
    tick = time.perf_counter(); groups, differences, physical = evaluate(root, config, law, fixture, seals)
    timings['evaluation_seconds'] = time.perf_counter()-tick
    report = {'status':'synthetic_scalar_filter_benchmark_completed', 'run_directory':str(root),
        'cases':len(cases), 'in_prior_cases':sum(c['in_prior'] for c in cases),
        'observations':len(json.loads((root/'observations.json').read_text())), 'groups':groups,
        'dynamic_audit':{k:v for k,v in audit.items() if k != 'records'}, 'timings':timings,
        'grid_refinement_mean_gap_p95_mm':float(np.quantile([r['mean_gap_difference_mm'] for r in differences], .95)),
        'grid_refinement_max_mean_gap_mm':max(r['mean_gap_difference_mm'] for r in differences),
        'grid_refinement_max_separation_mass_difference':max(r['separation_mass_difference'] for r in differences),
        'maximum_true_probe_extra_damage':max(r['extra_damage_over_repeated_weak'] for r in physical),
        'seals':seals, 'novel_method_validated':False, 'real_camera_gate_passed':False,
        'limitations':['Known scalar stable quasistatic law; observations are noisy hold gaps, not rendered or measured images.',
            'Dense grids approximate the specified uniform-parameter Bayesian posterior; they are not continuous certified bounds or true-state guarantees.',
            'Hard-filter support is a finite grid. Its uniform candidate fraction is not used as a calibrated event probability.',
            'Hard-filter arithmetic-mean forecasts are descriptive point summaries, not Bayesian posterior means; only full retained-support coverage is reported.',
            'SMC and resample-move are standard baselines; no new method is claimed. Three RNG replicates do not triple the number of independent specimens.',
            'Cases and protocol were designed after earlier pilot results; this is a development diagnostic, not an untouched confirmatory test.',
            'Noise levels and bias are assumed scalar measurement errors, not real camera performance. Constant bias is intentionally omitted from inference.',
            'Fixed probes may add damage; no active policy or safe-action guarantee is evaluated.',
            'Observed gaps come from the same analytical physics family used by inference. Dynamic audit is a numerical reduction check, not external physics validation.']}
    write_json(root/'report.json', report)
    out = project/'artifacts/synthetic-filter-benchmark'; out.mkdir(parents=True, exist_ok=True)
    for name in ('config.json','code-sha256.json','report.json','dynamic-audit.json','grid-refinement.json','intervention-costs.json','evaluation-rows.json','sealed-predictions.json'):
        shutil.copy2(root/name, out/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','limitations','seals')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
