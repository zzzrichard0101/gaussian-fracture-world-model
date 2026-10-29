"""Blind finite-catalog spatial observation and two-step policy development."""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from wmfracture.spatial_fixture import three_zone_fixture
from wmfracture.spatial_observation import PhysicalCatalog, infer_public, choose_public


POLICY_KEYS = ('zero_action', 'initial_weak_action', 'left_weak_action', 'right_weak_action',
    'left_mid_action', 'right_mid_action', 'tilt_threshold_sigma', 'bias_sigma_m', 'bias_bound_m',
    'noise_multiple', 'damage_cap')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def parameter_sets(settings):
    parameters = [list(p) for p in product(settings['energy_values_j_m2'], settings['preparation_values_m'], settings['preparation_values_m'])]
    for energy in (settings['energy_values_j_m2'][0], settings['energy_values_j_m2'][-1]):
        f = 2*energy/1000.; d = settings['matched_damage']
        h = f*.0001/(f-d*(f-.0001)); remote = h+(1-d)*.01*1e7*h/10000.
        parameters.append([energy, remote, remote])
    rng = np.random.default_rng(settings['stress_seed']); outside = []
    for _ in range(settings['outside_catalog_cases']):
        outside.append([float(rng.uniform(*settings['outside_energy_range'])),
            *map(float, rng.uniform(*settings['outside_preparation_range_m'], 2))])
    assert not any(np.allclose(a, b, rtol=0., atol=1e-12) for a in outside for b in parameters)
    return parameters, parameters+outside


def noise_array(seed, count, sigma, multiple):
    rng = np.random.default_rng(seed); values = []
    while len(values) < count*2:
        value = float(rng.normal())
        if abs(value) < multiple: values.append(sigma*value)
    return np.asarray(values).reshape(count, 2)


def public_forecast(catalog, path, posterior, initial):
    if posterior['weights'] is None: return None
    gaps, events = catalog.future(tuple(path)); w = posterior['weights']; mask = posterior['feasible']
    before = None if initial['weights'] is None else np.einsum('n,nij->ij', initial['weights'], gaps)
    return {'mean': np.einsum('n,nij->ij', w, gaps), 'before_mean': before,
        'event': w@np.any(events, axis=2), 'lower': gaps[mask].min(axis=0), 'upper': gaps[mask].max(axis=0)}


def collect(settings, catalog, truth, catalog_size):
    ps = {k: settings[k] for k in POLICY_KEYS}
    rows, public, decisions = [], [], []
    count = settings['additional_observations']+2
    for ci in range(len(truth.physics.parameters)):
        kind = 'catalog' if ci < catalog_size else 'outside'
        for si, sigma in enumerate(settings['sigma_m']):
            for bi, bias in enumerate(settings['actual_bias_vectors_m']):
                for repeat in range(settings['noise_repetitions']):
                    seed = settings['noise_seed']+ci*10000+si*100+bi*10+repeat
                    errors = noise_array(seed, count, sigma, settings['noise_multiple'])
                    context = f's-{ci}-{si}-{bi}-{repeat}'
                    for modality in settings['modalities']:
                        positions = modality['positions_m']
                        for policy in settings['policies']:
                            path, observed = [], []
                            def observe(action):
                                path.append(int(action))
                                prediction = truth.observations(tuple(path), positions)[ci]
                                observed.append((prediction+np.asarray(bias)+errors[len(path)-1]).tolist())
                            observe(ps['zero_action']); observe(ps['initial_weak_action'])
                            initial = infer_public(catalog, path, observed, positions, sigma, ps)
                            identifier = context+'-'+modality['name']+'-'+policy
                            for turn in range(settings['additional_observations']):
                                tick = time.perf_counter()
                                # No ci, kind, seed, true index, parameter vector or future labels crosses this call.
                                action, belief, log = choose_public(policy, catalog, path, observed, positions, sigma, turn, ps)
                                decisions.append({'trajectory_id': identifier, 'turn': turn,
                                    'public_actions': path.copy(), 'public_observed_m': [v.copy() for v in observed],
                                    'positions_m': positions, 'sigma_m': sigma, 'policy': policy,
                                    'action': action, 'feasible_count': int(belief['feasible'].sum()),
                                    'plan': log, 'seconds_with_shared_physics_cache': time.perf_counter()-tick})
                                if action is None: break
                                observe(action)
                            posterior = infer_public(catalog, path, observed, positions, sigma, ps)
                            forecast = public_forecast(catalog, path, posterior, initial)
                            target, events = truth.future(tuple(path)); target, events = target[ci], np.any(events[ci], axis=1)
                            risk = float(truth.damage_increment(tuple(path))[ci])
                            record = {'id': identifier, 'context_id': context, 'case_index': ci, 'case_kind': kind,
                                'sigma_m': sigma, 'bias_index': bi, 'repeat': repeat, 'modality': modality['name'], 'policy': policy,
                                'actions': path.copy(), 'delivered_frames': len(path), 'scalar_readouts': len(path)*len(positions),
                                'prediction_available': forecast is not None, 'finite_feasible_count': int(posterior['feasible'].sum()),
                                'true_candidate_retained': bool(posterior['feasible'][ci]) if kind == 'catalog' else None,
                                'actual_added_damage': risk, 'actual_damage_violation': risk > settings['damage_cap']+1e-10,
                                'actual_probe_separation': bool(truth.separated(tuple(path))[ci]),
                                'gap_mse_mm2': None, 'before_gap_mse_mm2': None, 'event_brier': None,
                                'simultaneous_envelope_coverage': None, 'mean_envelope_width_mm': None}
                            if forecast is not None:
                                record.update({'gap_mse_mm2': float(np.mean(((forecast['mean']-target)*1000)**2)),
                                    'before_gap_mse_mm2': None if forecast['before_mean'] is None else float(np.mean(((forecast['before_mean']-target)*1000)**2)),
                                    'event_brier': float(np.mean((forecast['event']-events)**2)),
                                    'simultaneous_envelope_coverage': bool(np.all(target >= forecast['lower']-1e-12) and np.all(target <= forecast['upper']+1e-12)),
                                    'mean_envelope_width_mm': float(np.mean(forecast['upper']-forecast['lower'])*1000)})
                            rows.append(record)
                            public.append({'id': identifier, 'actions': path.copy(), 'observed_m': [v.copy() for v in observed],
                                'positions_m': positions, 'sigma_m': sigma})
        print(f'SPATIAL_OBSERVATION_CASE {ci+1}/{len(truth.physics.parameters)}', flush=True)
    return rows, public, decisions


def summarize(rows):
    result = []
    keys = sorted(set((r['case_kind'], r['sigma_m'], r['modality'], r['policy']) for r in rows))
    for kind, sigma, modality, policy in keys:
        subset = [r for r in rows if (r['case_kind'], r['sigma_m'], r['modality'], r['policy']) == (kind, sigma, modality, policy)]
        valid = [r for r in subset if r['prediction_available']]
        result.append({'case_kind': kind, 'sigma_m': sigma, 'modality': modality, 'policy': policy,
            'trajectories': len(subset), 'predictions': len(valid),
            'damage_violations': sum(r['actual_damage_violation'] for r in subset),
            'probe_separations': sum(r['actual_probe_separation'] for r in subset),
            'budget_unfilled': sum(r['delivered_frames'] < 4 for r in subset),
            'true_candidate_losses': sum(r['true_candidate_retained'] is False for r in subset),
            'rmse_mm_on_available': float(np.sqrt(np.mean([r['gap_mse_mm2'] for r in valid]))) if valid else None,
            'before_rmse_mm_on_available': float(np.sqrt(np.mean([r['before_gap_mse_mm2'] for r in valid]))) if valid else None,
            'event_brier_on_available': float(np.mean([r['event_brier'] for r in valid])) if valid else None,
            'coverage_on_available': float(np.mean([r['simultaneous_envelope_coverage'] for r in valid])) if valid else None,
            'mean_envelope_width_mm_on_available': float(np.mean([r['mean_envelope_width_mm'] for r in valid])) if valid else None,
            'mean_added_damage': float(np.mean([r['actual_added_damage'] for r in subset]))})
    return result


def audit_reference(settings, truth):
    records = []
    for ci in settings['reference_audit_case_indices']:
        energy, left, right = truth.physics.parameters[ci]
        model = three_zone_fixture(float(energy)); state = model.virgin()
        model.cycle(state, [left, right], settings['reference_drive_increment_m'])
        history_error = float(np.max(abs(state.maximum_opening-truth.transition(())[ci])))
        path = []
        for action in [0, 1, 4, 5]:
            path.append(action); drive = settings['actions_m'][action]
            model.segment(state, drive, settings['reference_drive_increment_m'])
            truth.transition(tuple(path)); q = truth.peaks[tuple(path)][ci]
            gap_error = float(np.max(abs(model.B@(state.q-q))))
            model.segment(state, [0., 0.], settings['reference_drive_increment_m'])
            h = truth.transition(tuple(path))[ci]
            history_error = max(history_error, float(np.max(abs(state.maximum_opening-h))))
            damage_error = float(np.max(abs(model.damages(state.maximum_opening)-truth.physics.damage(truth.transition(tuple(path)))[ci])))
            records.append({'case_index': ci, 'path': path.copy(), 'gap_error_m': gap_error,
                'maximum_error_m': history_error, 'damage_error': damage_error})
    return records


def audit_refinement(settings, catalog, public, decisions):
    refined = PhysicalCatalog(catalog.physics.parameters, settings['actions_m'], settings['future_drives_m'], settings['refined_cycle_increments'])
    max_gap = max_history = max_mean = max_l1 = 0.; mask_changes = decision_changes = 0
    for path in list(catalog.states):
        refined.transition(path)
        max_history = max(max_history, float(np.max(abs(catalog.states[path]-refined.states[path]))))
        if path: max_gap = max(max_gap, float(np.max(abs(catalog.peaks[path]-refined.peaks[path]))))
    ps = {k: settings[k] for k in POLICY_KEYS}
    for i, episode in enumerate(public):
        args = (episode['actions'], episode['observed_m'], episode['positions_m'], episode['sigma_m'], ps)
        first, second = infer_public(catalog, *args), infer_public(refined, *args)
        mask_changes += int(not np.array_equal(first['feasible'], second['feasible']))
        if first['weights'] is not None and second['weights'] is not None:
            max_l1 = max(max_l1, float(np.sum(abs(first['weights']-second['weights']))))
            a = np.einsum('n,nij->ij', first['weights'], catalog.future(tuple(episode['actions']))[0])
            b = np.einsum('n,nij->ij', second['weights'], refined.future(tuple(episode['actions']))[0])
            max_mean = max(max_mean, float(np.max(abs(a-b))))
        elif (first['weights'] is None) != (second['weights'] is None): mask_changes += 1
    for decision in decisions:
        if decision['policy'] != 'finite_guard_tilt': continue
        action, _, _ = choose_public(decision['policy'], refined, decision['public_actions'], decision['public_observed_m'],
            decision['positions_m'], decision['sigma_m'], decision['turn'], ps)
        decision_changes += int(action != decision['action'])
    return {'max_peak_coordinate_change_m': max_gap, 'max_history_change_m': max_history,
        'max_posterior_mean_change_m': max_mean, 'max_weight_l1_change': max_l1,
        'changed_feasible_masks': mask_changes, 'changed_guard_actions_on_fixed_prefix': decision_changes,
        'scope': '64/128 increments of the same discrete catalog, not continuous parameter-grid convergence or an independent physical solver.'}


def figure(output, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [r for r in summaries if r['case_kind'] == 'catalog' and r['sigma_m'] == .00002]
    policies = ['rest_repeat', 'fixed_lr', 'fixed_rl', 'tilt_same_side', 'finite_guard_tilt']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(policies))
    for name, offset in [('center_pair', -.18), ('endpoints', .18)]:
        rows = [next(r for r in selected if r['modality'] == name and r['policy'] == p) for p in policies]
        axes[0].bar(x+offset, [r['rmse_mm_on_available'] or 0 for r in rows], .34, label=name)
        axes[1].bar(x+offset, [r['damage_violations']/r['trajectories'] for r in rows], .34, label=name)
    for axis in axes:
        axis.set_xticks(x); axis.set_xticklabels(policies, rotation=20, ha='right', fontsize=8); axis.legend(fontsize=8)
    axes[0].set_ylabel('Future-gap RMSE (mm), available predictions')
    axes[1].set_ylabel('Actual cumulative local-damage violation fraction')
    fig.suptitle('Finite 29-state catalog / 0.020mm noise: synthetic readouts, not images')
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='config/spatial-observation.json')
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    settings = json.loads((project/args.config).read_text())
    output = project/'artifacts/spatial-observation'; output.mkdir(parents=True, exist_ok=True)
    parameters, true_parameters = parameter_sets(settings)
    write(output/'config.json', settings); write(output/'case-config.json', {'catalog_parameters': parameters, 'true_parameters': true_parameters})
    write(output/'catalog.json', parameters)
    sources = ['wmfracture/spatial_observation.py', 'wmfracture/spatial_fixture.py', 'wmfracture/cohesive.py',
        'wmfracture/response_quadrature.py', 'scripts/spatial_observation_evaluate.py', 'tests/test_spatial_observation.py', args.config]
    write(output/'code-sha256.json', {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in sources})
    start = time.perf_counter()
    catalog = PhysicalCatalog(parameters, settings['actions_m'], settings['future_drives_m'], settings['cycle_increments'])
    truth = PhysicalCatalog(true_parameters, settings['actions_m'], settings['future_drives_m'], settings['cycle_increments'])
    rows, public, decisions = collect(settings, catalog, truth, len(parameters))
    write(output/'evaluation-rows.json', rows); write(output/'public-episodes.json', public); write(output/'decisions.json', decisions)
    summaries = summarize(rows); write(output/'analysis-summary.json', summaries)
    print('SPATIAL_OBSERVATION reference audit', flush=True)
    reference = audit_reference(settings, truth); write(output/'reference-audit.json', reference)
    print('SPATIAL_OBSERVATION refinement audit', flush=True)
    refinement = audit_refinement(settings, catalog, public, decisions); write(output/'refinement-audit.json', refinement)
    look = {r['id']: r for r in public}; control_pairs = []
    for row in public:
        if row['id'].endswith('-center_pair-fixed_lr'):
            other = look[row['id'][:-len('fixed_lr')]+'tilt_same_side']
            control_pairs.append(row['actions'] == other['actions'] and row['observed_m'] == other['observed_m'])
    thresholds = settings['thresholds']
    checks = {'finite_catalog_true_retention': all(r['true_candidate_retained'] for r in rows if r['case_kind'] == 'catalog'),
        'finite_guard_on_catalog': not any(r['actual_damage_violation'] or r['actual_probe_separation'] for r in rows if r['case_kind'] == 'catalog' and r['policy'] == 'finite_guard_tilt'),
        'center_control_exact_replay': all(control_pairs),
        'reference_gap': max(r['gap_error_m'] for r in reference) <= thresholds['gap_m'],
        'reference_history': max(r['maximum_error_m'] for r in reference) <= thresholds['maximum_m'],
        'reference_damage': max(r['damage_error'] for r in reference) <= thresholds['damage'],
        'refined_peak': refinement['max_peak_coordinate_change_m'] <= thresholds['gap_m'],
        'refined_history': refinement['max_history_change_m'] <= thresholds['maximum_m'],
        'refined_posterior': refinement['max_posterior_mean_change_m'] <= thresholds['posterior_mean_m'] and refinement['max_weight_l1_change'] <= thresholds['posterior_l1'],
        'refined_guard_actions': refinement['changed_guard_actions_on_fixed_prefix'] == 0}
    figure(output, summaries)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'settings': settings,
        'catalog_states': len(parameters), 'outside_states': len(true_parameters)-len(parameters), 'trajectories': len(rows),
        'decisions': len(decisions), 'center_control_pairs': len(control_pairs), 'checks': checks,
        'all_checks_passed': all(checks.values()), 'refinement': refinement, 'summaries': summaries,
        'elapsed_seconds': time.perf_counter()-start, 'scope': settings['scope'],
        'timing_scope': 'Physics caches are shared across trajectories; per-decision logs are NOT independent end-to-end policy timing benchmarks.'}
    write(output/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('settings', 'summaries')}, indent=2), flush=True)
    if not report['all_checks_passed']: raise SystemExit('Spatial observation gate failed; inspect saved evidence')


if __name__ == '__main__':
    main()
