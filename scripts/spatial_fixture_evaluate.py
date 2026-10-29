"""Three-zone forward mechanics and same-fixture preparation diagnostic."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from wmfracture.spatial_fixture import three_zone_fixture


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def preparation(case, model, damage):
    if case['preparation'] != 'uniform_matched':
        return case['preparation']
    law = replace(model.laws[0], area=sum(v.area for v in model.laws))
    h = law.failure_opening*law.initiation_opening/(law.failure_opening-damage*(law.failure_opening-law.initiation_opening))
    drive = h+law.area*law.traction(h, h)/sum(model.springs)
    return [float(drive)]*2


def execute(model, state, peak, resolution, reference_damage):
    trace, audit = model.cycle(state, peak, resolution)
    up = [r for r in trace if r['segment'] == 'up']
    hold = up[-1]
    max_added = max(float(np.max(np.asarray(row['damage'])-reference_damage)) for row in trace)
    max_error = max(abs(row['accounted_j']) for row in trace)
    max_angle = max(abs(row['rotation_rad']) for row in trace)
    fail = next(({'drive_m': row['drive_m'], 'zone_mask': [v == 1. for v in row['damage']]}
                 for row in up if any(v == 1. for v in row['damage'])), None)
    keep = np.unique(np.linspace(0, len(trace)-1, min(81, len(trace)), dtype=int))
    return {'peak': hold, 'final': model.snapshot(state), 'audit': audit,
        'max_added_damage_from_preparation': max_added, 'max_accounted_residual_j': max_error,
        'max_rotation_rad': max_angle, 'first_separation_sample': fail,
        'sampled_trace': [trace[i] for i in keep]}


def evaluate(settings):
    rows, summaries = [], []
    for ci, case in enumerate(settings['cases']):
        for resolution in settings['maximum_drive_increments_m']:
            model = three_zone_fixture(case['energy_j_m2'], settings['spring_each_n_m'])
            state = model.virgin(); peak = preparation(case, model, settings['matched_damage'])
            prepared = execute(model, state, peak, resolution, np.zeros(3))
            baseline = model.damages(state.maximum_opening)
            summaries.append({'case_id': case['id'], 'resolution_m': resolution, 'preparation_drive_m': peak,
                'minimum_global_tangent_bound_n_m': model.minimum_tangent_bound,
                'prepared': prepared})
            for kind, queries in (('probe', settings['probes']), ('future', settings['future_queries'])):
                for query in queries:
                    own = state.clone()
                    result = execute(model, own, query['drive_m'], resolution, baseline)
                    rows.append({'case_id': case['id'], 'resolution_m': resolution, 'kind': kind,
                        'query_id': query['id'], 'drive_m': query['drive_m'], **result})
            # Separate branch, but a single persistent state WITHIN this branch.
            own = state.clone()
            for turn, name in enumerate(settings['sequential_probes']):
                query = next(q for q in settings['probes'] if q['id'] == name)
                result = execute(model, own, query['drive_m'], resolution, baseline)
                rows.append({'case_id': case['id'], 'resolution_m': resolution, 'kind': 'sequential',
                    'query_id': name, 'turn': turn, 'drive_m': query['drive_m'], **result})
        print(f'SPATIAL_CASE {ci+1}/{len(settings["cases"])} {case["id"]}', flush=True)
    return rows, summaries


def analyze(settings, rows, preparations):
    resolutions = settings['maximum_drive_increments_m']; fine = min(resolutions); middle = sorted(resolutions)[1]
    def identity(r):
        return r['case_id'], r['kind'], r['query_id'], r.get('turn', 0)
    lookup = {(identity(r), r['resolution_m']): r for r in rows}
    refinements = []
    for row in rows:
        if row['resolution_m'] != fine: continue
        previous = lookup[(identity(row), middle)]
        refinements.append({'case_id': row['case_id'], 'kind': row['kind'], 'query_id': row['query_id'],
            'max_gap_change_m': float(np.max(abs(np.asarray(row['peak']['gaps_m'])-previous['peak']['gaps_m']))),
            'max_history_change_m': float(np.max(abs(np.asarray(row['final']['maximum_opening_m'])-previous['final']['maximum_opening_m']))),
            'max_damage_change': float(np.max(abs(np.asarray(row['final']['damage'])-previous['final']['damage']))),
            'separation_masks_equal': [d == 1. for d in row['final']['damage']] == [d == 1. for d in previous['final']['damage']]})
    def get(case, query):
        return next(r for r in rows if r['case_id'] == case and r['resolution_m'] == fine and r['query_id'] == query and r['kind'] == 'probe')
    matched = []
    for query in settings['probes']:
        a, b = (get(case, query['id']) for case in ('matched_low_energy', 'matched_high_energy'))
        matched.append({'query_id': query['id'],
            'max_gap_difference_m': float(np.max(abs(np.asarray(a['peak']['gaps_m'])-b['peak']['gaps_m']))),
            'center_gap_difference_m': abs(a['peak']['gaps_m'][1]-b['peak']['gaps_m'][1]),
            'max_added_damage_both': max(a['max_added_damage_from_preparation'], b['max_added_damage_from_preparation']),
            'pair_only_within_diagnostic_cap': max(a['max_added_damage_from_preparation'], b['max_added_damage_from_preparation']) <= settings['diagnostic_damage_cap']+1e-12})
    mirror_errors = []
    for left, right in (('center_weak', 'center_weak'), ('left_mid', 'right_mid'), ('left_strong', 'right_strong')):
        a, b = get('left_prepared', left), get('right_prepared', right)
        mirror_errors.append(float(np.max(abs(np.asarray(a['peak']['gaps_m'])-np.asarray(b['peak']['gaps_m'])[::-1]))))
    all_records = rows+[p['prepared'] for p in preparations]
    t = settings['thresholds']
    metrics = {'max_force_residual_n': max(r['audit']['max_force_residual_n'] for r in all_records),
        'max_moment_residual_nm': max(r['audit']['max_moment_residual_nm'] for r in all_records),
        'max_unloaded_gap_m': max(max(abs(x) for x in r['final']['gaps_m']) for r in all_records),
        'max_accounted_residual_j': max(r['max_accounted_residual_j'] for r in all_records),
        'max_rotation_rad': max(r['max_rotation_rad'] for r in all_records),
        'max_finest_gap_change_m': max(r['max_gap_change_m'] for r in refinements),
        'max_finest_history_change_m': max(r['max_history_change_m'] for r in refinements),
        'max_finest_damage_change': max(r['max_damage_change'] for r in refinements),
        'max_mirror_error_m': max(mirror_errors),
        'minimum_global_tangent_bound_n_m': min(p['minimum_global_tangent_bound_n_m'] for p in preparations)}
    checks = {'force_balance': metrics['max_force_residual_n'] <= t['force_residual_n'],
        'moment_balance': metrics['max_moment_residual_nm'] <= t['moment_residual_nm'],
        'complete_unloading': metrics['max_unloaded_gap_m'] <= t['unloaded_gap_m'],
        'work_balance': metrics['max_accounted_residual_j'] <= t['accounted_residual_j'],
        'small_rotation': metrics['max_rotation_rad'] <= t['maximum_rotation_rad'],
        'displacement_refinement': metrics['max_finest_gap_change_m'] <= t['finest_pair_gap_m'],
        'history_refinement': metrics['max_finest_history_change_m'] <= t['finest_pair_maximum_m'],
        'damage_refinement': metrics['max_finest_damage_change'] <= t['finest_pair_damage'],
        'separation_refinement': all(r['separation_masks_equal'] for r in refinements),
        'mirror_symmetry': metrics['max_mirror_error_m'] <= t['symmetry_and_matched_gap_m'],
        'matched_weak_observations': all(r['max_gap_difference_m'] <= t['symmetry_and_matched_gap_m'] for r in matched if 'weak' in r['query_id']),
        'stable_incremental_mechanics': metrics['minimum_global_tangent_bound_n_m'] > 0.}
    return {'metrics': metrics, 'checks': checks, 'all_checks_passed': all(checks.values()),
            'refinements': refinements, 'matched_pair': matched,
            'scope': 'Five designed forward cases; branch copies are not sequentially available measurements. Pair damage flags are not full-uncertainty safety certification. Refinement is empirical, not a rigorous path error bound.'}


def figures(output, settings, rows, preparations, analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fine = min(settings['maximum_drive_increments_m'])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.array([-.04, 0., .04])*1000
    for name in ('left_prepared', 'right_prepared'):
        p = next(p for p in preparations if p['case_id'] == name and p['resolution_m'] == fine)
        axes[0].plot(x, p['prepared']['final']['damage'], 'o-', label=name)
    axes[0].set(xlabel='Interface position (mm)', ylabel='Prepared local damage D', ylim=(0., 1.02)); axes[0].legend(fontsize=8)
    for case in ('matched_low_energy', 'matched_high_energy'):
        for query, style in (('left_weak', '--'), ('left_mid', '-')):
            r = next(r for r in rows if r['case_id'] == case and r['resolution_m'] == fine and r['query_id'] == query and r['kind'] == 'probe')
            axes[1].plot(x, np.asarray(r['peak']['gaps_m'])*1000, 'o'+style, label=f'{case}, {query}')
    axes[1].set(xlabel='Interface position (mm)', ylabel='Peak local gap (mm)'); axes[1].legend(fontsize=6)
    for query in ('left_mid', 'left_strong'):
        yy = []
        for step in settings['maximum_drive_increments_m']:
            r = next(r for r in rows if r['case_id'] == 'matched_low_energy' and r['resolution_m'] == step and r['query_id'] == query and r['kind'] == 'probe')
            yy.append(r['max_accounted_residual_j'])
        axes[2].plot(np.array(settings['maximum_drive_increments_m'])*1e6, yy, 'o-', label=query)
    axes[2].set(xlabel='Maximum drive increment (micrometers)', ylabel='Max work-balance residual (J)'); axes[2].legend(fontsize=8)
    fig.suptitle('Three-zone forward diagnostic: physical histories, spatial response, refinement')
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='config/spatial-fixture.json')
    args = parser.parse_args(); project = Path(__file__).resolve().parents[1]
    settings = json.loads((project/args.config).read_text())
    output = project/'artifacts/spatial-fixture'; output.mkdir(parents=True, exist_ok=True)
    sources = ['wmfracture/spatial_fixture.py', 'wmfracture/cohesive.py',
        'scripts/spatial_fixture_evaluate.py', 'tests/test_spatial_fixture.py', args.config]
    write(output/'config.json', settings)
    write(output/'code-sha256.json', {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in sources})
    start = time.perf_counter(); rows, preparations = evaluate(settings)
    analysis = analyze(settings, rows, preparations)
    write(output/'analysis-summary.json', analysis)
    write(output/'evaluation-rows.json', [{k:v for k,v in r.items() if k != 'sampled_trace'} for r in rows])
    write(output/'preparations.json', [{k:v if k != 'prepared' else {a:b for a,b in v.items() if a != 'sampled_trace'} for k,v in p.items()} for p in preparations])
    write(output/'traces.json', rows)
    figures(output, settings, rows, preparations, analysis)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'settings': settings,
        'branch_cycles': len(rows), 'preparation_cycles': len(preparations),
        'scope': settings['scope'], 'elapsed_seconds': time.perf_counter()-start,
        'analysis': analysis}
    write(output/'report.json', report)
    print(json.dumps({'output': str(output), 'branch_cycles': len(rows), 'analysis': {k:v for k,v in analysis.items() if k != 'refinements'}}, indent=2), flush=True)
    if not analysis['all_checks_passed']:
        raise SystemExit('Spatial forward gate failed; inspect the saved evidence before extending the pipeline')


if __name__ == '__main__':
    main()
