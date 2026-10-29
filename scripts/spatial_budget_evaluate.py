"""Nested total-physics-budget comparison; predictions freeze before truth reads."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from wmfracture.spatial_budget import (path_endpoint_cost, main_allowance, physical_pilot_proposal,
    equal_hybrid, select_physics_draws, cutoff_for_allowance)
from wmfracture.spatial_observation import PhysicalCatalog
from wmfracture.spatial_transport import PublicForward, bounded_loglike, fit_proposals, summarize_importance
from spatial_observation_evaluate import POLICY_KEYS, write


def main():
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project/'config/spatial-budget.json').read_text())
    proposal_config = json.loads((project/config['proposal_config']).read_text())
    settings = json.loads((project/'docs/results/spatial-observation-config.json').read_text())
    episodes = json.loads((project/config['public_inputs']).read_text())
    output = project/'artifacts/spatial-budget'; output.mkdir(parents=True, exist_ok=True)
    write(output/'config.json', config); write(output/'proposal-config.json', proposal_config)
    write(output/'inputs.json', episodes)
    source_paths = ['config/spatial-budget.json', config['proposal_config'], 'wmfracture/spatial_budget.py',
        'wmfracture/spatial_transport.py', 'wmfracture/spatial_observation.py',
        'wmfracture/response_quadrature.py', 'scripts/spatial_budget_evaluate.py', 'tests/test_spatial_budget.py']
    write(output/'code-sha256.json', {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in source_paths})
    input_paths = [config['public_inputs'], 'docs/results/spatial-observation-config.json']
    write(output/'source-sha256.json', {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in input_paths})
    bounds = np.asarray(proposal_config['parameter_bounds'])
    ps = {k: settings[k] for k in POLICY_KEYS}
    budgets = sorted(config['endpoint_budgets']); predictions = []; costs = []; adaptation = []
    overall = time.perf_counter()
    for replica, (pilot_seed, seed) in enumerate(zip(config['pilot_seeds'], config['scramble_seeds'])):
        methods = config['methods'][replica:]+config['methods'][:replica]
        for ei, episode in enumerate(episodes):
            prefix = episode['actions']; observed = episode['observed_m']; sigma = episode['sigma_m']
            unit_cost = path_endpoint_cost(settings['actions_m'], settings['future_drives_m'], prefix,
                proposal_config['cycle_increments'])
            for method in methods:
                started = time.perf_counter()
                # Each method gets a fresh bank; common random numbers share no physics cache.
                forward = PublicForward(bounds, settings['actions_m'], settings['future_drives_m'],
                    episode['positions_m'], proposal_config['cycle_increments'])
                proposal = None; audit = None; pilot_cost = 0; pilot_nodes = 0
                if method != 'prior_qmc':
                    pilot_unit = qmc.Sobol(3, scramble=True, seed=pilot_seed).random_base2(proposal_config['pilot_power'])
                    pilot_prediction, _ = forward.evaluate(pilot_unit, prefix)
                    pilot_cost = forward.endpoint_parameter_evaluations; pilot_nodes = len(pilot_unit)
                    if method == 'physical_mixture':
                        proposal = physical_pilot_proposal(pilot_unit, pilot_prediction, observed, sigma, ps, proposal_config)
                    else:
                        proposals, audit = fit_proposals(forward, pilot_unit, pilot_prediction, prefix, observed, sigma, ps, proposal_config)
                        proposal = proposals['response_chart'] if method == 'response_chart' else equal_hybrid(
                            proposals['projected_physical'], proposals['response_chart'])
                setup = int(forward.endpoint_parameter_evaluations)
                setup_parameters = forward.parameter_evaluations
                allowances = [main_allowance(b, setup, unit_cost) for b in budgets]
                unit4 = qmc.Sobol(4, scramble=True, seed=seed).random_base2(config['maximum_draw_power'])
                points, logq, inside = select_physics_draws(proposal, unit4, allowances[-1], config['draw_chunk'])
                fitted, future = forward.evaluate(points[inside], prefix, future=True)
                assert forward.endpoint_parameter_evaluations-setup == int(inside.sum())*unit_cost
                loglike = np.full(len(points), -np.inf)
                loglike[inside] = bounded_loglike(fitted, observed, sigma, ps)
                gaps = np.zeros((len(points), *future[0].shape[1:])); events = np.zeros_like(gaps, bool)
                gaps[inside], events[inside] = future
                elapsed = time.perf_counter()-started
                adaptation.append({'id': episode['id'], 'method': method, 'replica': replica, 'audit': audit,
                    'centers_unit': None if proposal is None else proposal.centers.tolist(),
                    'covariances_unit': None if proposal is None else proposal.covariances.tolist()})
                for budget, count in zip(budgets, allowances):
                    end = cutoff_for_allowance(inside, count)
                    result = summarize_importance(loglike[:end], logq[:end], gaps[:end], events[:end])
                    used = setup+count*unit_cost
                    assert 0 <= budget-used < unit_cost
                    predictions.append({'id': episode['id'], 'method': method, 'replica': replica,
                        'sigma_m': sigma, 'budget': budget, **result})
                    costs.append({'id': episode['id'], 'method': method, 'replica': replica, 'sigma_m': sigma,
                        'budget': budget, 'pilot_nodes': pilot_nodes, 'pilot_endpoint_cost': pilot_cost,
                        'adaptation_endpoint_cost': setup-pilot_cost,
                        'setup_parameter_evaluations': setup_parameters, 'setup_endpoint_cost': setup,
                        'full_path_endpoint_cost_per_parameter': unit_cost,
                        'main_parameter_evaluations': count, 'main_endpoint_cost': count*unit_cost,
                        'used_endpoint_cost': used, 'unused_endpoint_budget': budget-used,
                        'proposal_draws': end, 'outside_cube_draws': end-count,
                        'measured_maximum_budget_seconds': elapsed if budget == budgets[-1] else None,
                        'cost_verified_against_actual_maximum_run': True})
                print(f'SPATIAL_BUDGET replica={replica} history={ei+1}/{len(episodes)} method={method} '
                    f'used={costs[-1]["used_endpoint_cost"]}/{budgets[-1]} seconds={elapsed:.2f}', flush=True)
    write(output/'predictions.json', predictions); write(output/'costs.json', costs)
    write(output/'adaptation.json', adaptation)
    frozen = hashlib.sha256((output/'predictions.json').read_bytes()).hexdigest()
    # Evaluation-only inputs are opened after predictions have been saved and hashed.
    case_path = project/'docs/results/spatial-observation-case-config.json'
    old_path = project/'docs/results/spatial-observation-evaluation-rows.json'
    parameters = json.loads(case_path.read_text())['true_parameters']
    old = {r['id']: r for r in json.loads(old_path.read_text())}
    write(output/'evaluation-source-sha256.json', {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (case_path, old_path)})
    truth = PhysicalCatalog(parameters, settings['actions_m'], settings['future_drives_m'], proposal_config['cycle_increments'])
    lookup = {(r['id'], r['method'], r['replica'], r['budget']): r for r in predictions}
    evaluated = []
    for episode in episodes:
        target, _ = truth.future(tuple(episode['actions'])); target = target[old[episode['id']]['case_index']]
        for method in config['methods']:
            rows = [lookup[(episode['id'], method, replica, budget)] for replica in range(len(config['scramble_seeds'])) for budget in budgets[-2:]]
            available = all(r['available'] for r in rows); mean_range = event_range = None
            if available:
                mean_range = float(np.max(np.ptp(np.asarray([r['mean_gap_m'] for r in rows]), axis=0)))
                event_range = float(np.max(np.ptp(np.asarray([r['event_probability'] for r in rows]), axis=0)))
            minimum_ess = min(r['ess'] for r in rows)
            passed = bool(available and minimum_ess >= config['screen']['minimum_ess']
                and mean_range <= config['screen']['maximum_mean_change_m']
                and event_range <= config['screen']['maximum_event_change'])
            for budget in budgets:
                primary = lookup[(episode['id'], method, config['primary_replica'], budget)]
                evaluated.append({'id': episode['id'], 'method': method, 'sigma_m': episode['sigma_m'], 'budget': budget,
                    'available': primary['available'], 'ess': primary['ess'], 'screen_passed': passed,
                    'minimum_screen_ess': minimum_ess, 'max_mean_range_m': mean_range, 'max_event_range': event_range,
                    'mse_mm2': float(np.mean(((np.asarray(primary['mean_gap_m'])-target)*1000)**2)) if primary['available'] else None})
    summaries = []
    for sigma in settings['sigma_m']:
        for method in config['methods']:
            for budget in budgets:
                group = [r for r in evaluated if r['sigma_m'] == sigma and r['method'] == method and r['budget'] == budget]
                valid = [r for r in group if r['available']]
                resource = [r for r in costs if r['sigma_m'] == sigma and r['method'] == method and r['budget'] == budget]
                times = [r['measured_maximum_budget_seconds'] for r in resource if r['measured_maximum_budget_seconds'] is not None]
                summaries.append({'sigma_m': sigma, 'method': method, 'budget': budget,
                    'histories': len(group), 'available': len(valid), 'screen_passed_across_both_budgets': sum(r['screen_passed'] for r in group),
                    'median_ess': float(np.median([r['ess'] for r in group])),
                    'rmse_mm_on_available': float(np.sqrt(np.mean([r['mse_mm2'] for r in valid]))) if valid else None,
                    'main_physics_nodes_range': [min(r['main_parameter_evaluations'] for r in resource), max(r['main_parameter_evaluations'] for r in resource)],
                    'used_endpoint_cost_range': [min(r['used_endpoint_cost'] for r in resource), max(r['used_endpoint_cost'] for r in resource)],
                    'median_maximum_budget_seconds': float(np.median(times)) if times else None})
    write(output/'evaluation-rows.json', evaluated); write(output/'analysis-summary.json', summaries)
    report = {'config': config, 'histories': len(episodes), 'prediction_rows': len(predictions), 'summaries': summaries,
        'frozen_prediction_sha256': frozen, 'elapsed_seconds': time.perf_counter()-overall,
        'scope': config['scope'], 'accounting': config['accounting'], 'stopping': config['stopping']}
    write(output/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
