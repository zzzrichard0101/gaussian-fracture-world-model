"""Continuous-prior re-estimation on fixed public spatial histories."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from wmfracture.spatial_continuous import estimate
from wmfracture.spatial_observation import PhysicalCatalog
from spatial_observation_evaluate import POLICY_KEYS, write


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='config/spatial-continuous.json')
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    source = project/'docs/results'; output = project/'artifacts/spatial-continuous'; output.mkdir(parents=True, exist_ok=True)
    config = json.loads((project/args.config).read_text())
    settings = json.loads((source/'spatial-observation-config.json').read_text())
    old_rows = json.loads((source/'spatial-observation-evaluation-rows.json').read_text())
    public = json.loads((source/'spatial-observation-public-episodes.json').read_text())
    selected = {r['id']: r for r in old_rows if r['case_kind'] == 'outside' and r['modality'] == 'endpoints' and r['policy'] == 'finite_guard_tilt'}
    episodes = [r for r in public if r['id'] in selected]
    ps = {k: settings[k] for k in POLICY_KEYS}; bounds = np.asarray(config['parameter_bounds'])
    write(output/'config.json', config); write(output/'inputs.json', episodes)
    paths = ['wmfracture/spatial_continuous.py', 'scripts/spatial_continuous_evaluate.py',
        'wmfracture/spatial_observation.py', 'wmfracture/response_quadrature.py', args.config]
    write(output/'code-sha256.json', {p:hashlib.sha256((project/p).read_bytes()).hexdigest() for p in paths})
    write(output/'source-sha256.json', {n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in
        ('spatial-observation-config.json', 'spatial-observation-public-episodes.json', 'spatial-observation-evaluation-rows.json')})
    predictions = []; started = time.perf_counter()
    for seed in config['scramble_seeds']:
        unit = qmc.Sobol(3, scramble=True, seed=seed).random_base2(max(config['nested_powers']))
        parameters = bounds[:, 0]+unit*np.ptp(bounds, axis=1)
        bank = PhysicalCatalog(parameters, settings['actions_m'], settings['future_drives_m'], config['cycle_increments'])
        for i, episode in enumerate(episodes):
            path = tuple(episode['actions']); positions = episode['positions_m']
            prediction = np.stack([bank.observations(path[:j+1], positions) for j in range(len(path))], axis=1)
            gaps, events = bank.future(path)
            for power in config['nested_powers']:
                n = 2**power
                result = estimate(prediction[:n], gaps[:n], events[:n], episode['observed_m'], episode['sigma_m'], ps)
                predictions.append({'id':episode['id'], 'sigma_m':episode['sigma_m'], 'seed':seed, 'power':power, **result})
            print(f'SPATIAL_CONTINUOUS seed={seed} episode={i+1}/{len(episodes)}', flush=True)
    # Predictions are frozen before reading true parameter vectors or computing
    # any target future. Old selection metadata is post-hoc, as declared above.
    write(output/'predictions.json', predictions)
    frozen_hash = hashlib.sha256((output/'predictions.json').read_bytes()).hexdigest()
    truth_parameters = json.loads((source/'spatial-observation-case-config.json').read_text())['true_parameters']
    truth = PhysicalCatalog(truth_parameters, settings['actions_m'], settings['future_drives_m'], config['cycle_increments'])
    look = {(r['id'], r['seed'], r['power']):r for r in predictions}
    max_power = max(config['nested_powers']); previous_power = sorted(config['nested_powers'])[-2]
    evaluated = []
    for episode in episodes:
        comparison = [look[(episode['id'], seed, p)] for seed in config['scramble_seeds'] for p in (previous_power, max_power)]
        available = all(r['available'] for r in comparison)
        mean_change = event_change = None
        if available:
            means = np.asarray([r['mean_gap_m'] for r in comparison]); events = np.asarray([r['event_probability'] for r in comparison])
            mean_change = float(np.max(np.ptp(means, axis=0))); event_change = float(np.max(np.ptp(events, axis=0)))
        screened = bool(available and min(r['ess'] for r in comparison) >= config['screen']['minimum_ess']
            and mean_change <= config['screen']['maximum_mean_change_m'] and event_change <= config['screen']['maximum_event_change'])
        row = selected[episode['id']]; ci = row['case_index']; path = tuple(episode['actions'])
        target, events = truth.future(path); target, event = target[ci], np.any(events[ci], axis=1)
        primary = look[(episode['id'], config['scramble_seeds'][0], max_power)]
        mse = coverage = None
        if primary['available']:
            mse = float(np.mean(((np.asarray(primary['mean_gap_m'])-target)*1000)**2))
            coverage = bool(np.all(target >= np.asarray(primary['sample_envelope_lower_m'])-1e-12)
                and np.all(target <= np.asarray(primary['sample_envelope_upper_m'])+1e-12))
        evaluated.append({'id':episode['id'], 'sigma_m':episode['sigma_m'], 'primary_available':primary['available'],
            'primary_ess':primary['ess'], 'primary_mse_mm2':mse, 'sample_envelope_coverage':coverage,
            'finite_catalog_available':row['prediction_available'], 'finite_catalog_mse_mm2':row['gap_mse_mm2'],
            'screen_passed':screened, 'max_mean_range_m':mean_change, 'max_event_range':event_change})
    write(output/'evaluation-rows.json', evaluated)
    summaries = []
    for sigma in settings['sigma_m']:
        group = [r for r in evaluated if r['sigma_m'] == sigma]
        valid = [r for r in group if r['primary_available']]; stable = [r for r in group if r['screen_passed']]
        summaries.append({'sigma_m':sigma, 'histories':len(group), 'qmc_available':len(valid), 'screen_passed':len(stable),
            'qmc_rmse_mm_on_available':float(np.sqrt(np.mean([r['primary_mse_mm2'] for r in valid]))) if valid else None,
            'qmc_rmse_mm_on_screened':float(np.sqrt(np.mean([r['primary_mse_mm2'] for r in stable]))) if stable else None,
            'qmc_sample_coverage_on_available':float(np.mean([r['sample_envelope_coverage'] for r in valid])) if valid else None,
            'median_primary_ess':float(np.median([r['primary_ess'] for r in group]))})
    report = {'config':config, 'histories':len(episodes), 'qmc_rows':len(predictions), 'summaries':summaries,
        'frozen_prediction_sha256':frozen_hash, 'elapsed_seconds':time.perf_counter()-started,
        'scope':config['scope'], 'comparison_limit':'The continuous uniform prior differs from the old discrete prior. Fixed executed histories are unchanged; no new policy/safety improvement is inferred from re-estimation.'}
    write(output/'report.json', report); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
