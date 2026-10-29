"""Post-hoc public-input-only search for finite-guard counterexamples."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from wmfracture.spatial_fixture import three_zone_fixture
from wmfracture.spatial_observation import PhysicalCatalog, infer_public
from spatial_observation_evaluate import POLICY_KEYS, write


def independent_point(parameters, prefix, action, observed, positions, sigma, policy, settings, resolution):
    e, left, right = parameters; model = three_zone_fixture(float(e)); state = model.virgin()
    model.cycle(state, [left, right], resolution)
    baseline = model.damages(state.maximum_opening)
    prediction = []
    for index in prefix:
        model.segment(state, settings['actions_m'][index], resolution)
        prediction.append((state.q[0]+np.asarray(positions)*state.q[1]/.04).tolist())
        model.segment(state, [0., 0.], resolution)
    residual = np.asarray(observed)-prediction
    lower = np.maximum(-policy['bias_bound_m'], np.max(residual-policy['noise_multiple']*sigma, axis=0))
    upper = np.minimum(policy['bias_bound_m'], np.min(residual+policy['noise_multiple']*sigma, axis=0))
    current_damage = model.damages(state.maximum_opening)
    before_risk = float(np.max(current_damage-baseline))
    model.cycle(state, settings['actions_m'][action], resolution)
    risk = float(np.max(model.damages(state.maximum_opening)-baseline))
    return {'parameters': list(map(float, parameters)), 'resolution_m': resolution,
        'predicted_prefix_m': prediction, 'bias_lower_m': lower.tolist(), 'bias_upper_m': upper.tolist(),
        'minimum_bias_interval_width_m': float(np.min(upper-lower)), 'damage_increment': risk,
        'damage_increment_before_action': before_risk,
        'new_budget_crossing': bool(before_risk <= policy['damage_cap']+1e-12 and risk > policy['damage_cap']),
        'reference_damage': baseline.tolist(), 'final_damage': model.damages(state.maximum_opening).tolist(),
        'accounted_residual_j': model.energies(state)['accounted_j']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='config/spatial-failure-search.json')
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    source = project/'artifacts/spatial-observation'; output = project/'artifacts/spatial-failure-search'; output.mkdir(parents=True, exist_ok=True)
    config = json.loads((project/args.config).read_text()); settings = json.loads((source/'config.json').read_text())
    public = json.loads((source/'decisions.json').read_text()); rows = json.loads((source/'evaluation-rows.json').read_text())
    selected = {r['id'] for r in rows if r['case_kind'] == 'outside' and r['modality'] == 'endpoints'
        and r['sigma_m'] == .00002 and r['policy'] == 'finite_guard_tilt' and r['actual_damage_violation']}
    inputs = [{k:r[k] for k in ('trajectory_id', 'turn', 'public_actions', 'public_observed_m', 'positions_m', 'sigma_m', 'action')}
        for r in public if r['trajectory_id'] in selected and r['action'] not in (None, 0)]
    bounds = np.asarray(config['parameter_bounds'])
    unit = qmc.Sobol(3, scramble=True, seed=config['seed']).random_base2(config['sobol_power'])
    parameters = bounds[:, 0]+unit*np.ptp(bounds, axis=1)
    bank = PhysicalCatalog(parameters, settings['actions_m'], settings['future_drives_m'], config['cycle_increments'])
    ps = {k:settings[k] for k in POLICY_KEYS}; results = []
    for row in inputs:
        posterior = infer_public(bank, row['public_actions'], row['public_observed_m'], row['positions_m'], row['sigma_m'], ps)
        path = tuple(row['public_actions'])+(row['action'],)
        risk = bank.damage_increment(path)
        current_risk = bank.damage_increment(tuple(row['public_actions']))
        index = np.flatnonzero(posterior['feasible'] & (risk > ps['damage_cap']+config['minimum_damage_margin']))
        index = sorted(index, key=lambda i: (current_risk[i] > ps['damage_cap']+1e-12, -risk[i]))
        witness = None
        for i in index[:20]:
            audits = [independent_point(parameters[i], row['public_actions'], row['action'], row['public_observed_m'],
                row['positions_m'], row['sigma_m'], ps, settings, h) for h in config['reference_drive_increments_m']]
            if all(a['minimum_bias_interval_width_m'] >= config['minimum_bias_interval_width_m']
                    and a['damage_increment'] > ps['damage_cap']+config['minimum_damage_margin'] for a in audits):
                witness = {'parameter_point': parameters[i].tolist(), 'independent_replays': audits}; break
        results.append({'input': row, 'sampled_feasible_points': int(posterior['feasible'].sum()),
            'sampled_feasible_violating_points': len(index), 'witness': witness,
            'conclusion': 'numerical_counterexample_found' if witness else 'no_counterexample_found_not_an_allow'})
        print(f'SPATIAL_FAILURE_SEARCH {len(results)}/{len(inputs)} {results[-1]["conclusion"]}', flush=True)
    write(output/'config.json', config); write(output/'inputs.json', inputs); write(output/'witnesses.json', results)
    paths = ['scripts/spatial_failure_search.py', 'scripts/spatial_observation_evaluate.py',
        'wmfracture/spatial_observation.py', 'wmfracture/spatial_fixture.py', 'wmfracture/cohesive.py',
        'wmfracture/response_quadrature.py', args.config]
    write(output/'code-sha256.json', {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in paths})
    write(output/'source-sha256.json', {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('config.json', 'evaluation-rows.json', 'decisions.json')})
    report = {'config': config, 'selected_failed_trajectories': len(selected), 'public_queries': len(inputs),
        'sobol_points': len(parameters), 'counterexample_queries': sum(r['witness'] is not None for r in results),
        'trajectories_with_counterexamples': len({r['input']['trajectory_id'] for r in results if r['witness'] is not None}),
        'new_budget_crossing_queries': sum(r['witness'] is not None and r['witness']['independent_replays'][-1]['new_budget_crossing'] for r in results),
        'scope': config['scope']}
    write(output/'report.json', report); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
