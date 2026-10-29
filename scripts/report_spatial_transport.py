"""Derived diagnostics and figure; never change frozen transport estimates."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spatial_observation_evaluate import write


def main():
    project = Path(__file__).resolve().parents[1]
    root = project/'artifacts/spatial-transport'
    report = json.loads((root/'report.json').read_text())
    adaptation = json.loads((root/'adaptation.json').read_text())
    evaluated = json.loads((root/'evaluation-rows.json').read_text())
    predictions = json.loads((root/'predictions.json').read_text())
    config = report['config']
    fitted = [r for r in adaptation if 'audit' in r]
    cases = []
    for row in evaluated:
        if row['method'] != 'response_chart':
            continue
        runs = [r for r in predictions if r['id'] == row['id'] and r['method'] == row['method']
                and r['power'] in sorted(config['nested_powers'])[-2:]]
        cases.append({**row, 'minimum_screen_ess': min(r['ess'] for r in runs),
            'failed_screens': [name for name, failed in (
                ('available', not all(r['available'] for r in runs)),
                ('ess', min(r['ess'] for r in runs) < config['screen']['minimum_ess']),
                ('mean_change', row['max_mean_range_m'] is None or row['max_mean_range_m'] > config['screen']['maximum_mean_change_m']),
                ('event_change', row['max_event_range'] is None or row['max_event_range'] > config['screen']['maximum_event_change'])) if failed]})
    costs = []
    for method in config['methods']:
        group = [r for r in adaptation if r.get('method') == method]
        costs.append({'method': method,
            'fresh_main_parameter_evaluations_range': [min(r['main_parameter_evaluations'] for r in group), max(r['main_parameter_evaluations'] for r in group)],
            'fresh_main_endpoint_parameter_evaluations_range': [min(r['main_endpoint_parameter_evaluations'] for r in group), max(r['main_endpoint_parameter_evaluations'] for r in group)],
            'outside_cube_draws_range': [min(r['outside_cube_draws_at_maximum'] for r in group), max(r['outside_cube_draws_at_maximum'] for r in group)]})
    diagnostics = {'response_cases': cases, 'costs': costs,
        'adaptation_parameter_evaluations_per_fit': sorted(set(r['adaptation_parameter_evaluations'] for r in fitted)),
        'adaptation_endpoint_parameter_evaluations_range': [min(r['adaptation_endpoint_parameter_evaluations'] for r in fitted), max(r['adaptation_endpoint_parameter_evaluations'] for r in fitted)],
        'shared_pilot_nodes_per_replica': 2**config['pilot_power'],
        'shared_prior_nodes_per_replica': 2**max(config['nested_powers']),
        'rank_counts_all_fits': {str(i): sum(r['audit']['rank_counts'][str(i)] for r in fitted) for i in range(4)},
        'notes': ['Zero fresh main evaluations for prior_qmc means its physical bank was already computed and cached, not zero total cost.',
            'Fresh main counts exclude pilot, shared uniform bank and proposal adaptation. Endpoint counts count individual parameter/end-step solves, not vectorized Python calls.',
            'Screened convergence is an empirical screen of two replicas and two final levels; neither posterior accuracy nor a risk bound is established.']}
    write(root/'diagnostics.json', diagnostics)
    provenance = ['scripts/report_spatial_transport.py', 'docs/results/spatial-observation-case-config.json']
    write(root/'report-source-sha256.json', {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in provenance})
    methods = config['methods']; labels = ['Prior QMC', 'Physical mix', 'Projected mix', 'Response chart']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(methods)); width = .36
    for index, sigma in enumerate((1e-6, 2e-5)):
        rows = [next(r for r in report['summaries'] if r['method'] == method and r['sigma_m'] == sigma) for method in methods]
        at = x+(index-.5)*width
        label = f'sigma={sigma*1000:g} mm'
        axes[0].bar(at, [r['median_ess'] for r in rows], width, label=label)
        axes[1].bar(at, [r['screen_passed'] for r in rows], width, label=label)
        axes[2].bar(at, [r['rmse_mm_on_available'] for r in rows], width, label=label)
        for pos, row in zip(at, rows):
            axes[2].annotate(f"{row['available']}/4", (pos, row['rmse_mm_on_available']), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Median ESS, primary replica (log scale)')
    axes[0].axhline(40, color='black', linewidth=.8, linestyle='--', label='ESS threshold')
    axes[1].set(ylabel='Histories passing all numerical screens', ylim=(0,4.4), yticks=range(5))
    axes[2].set(ylabel='Future gap RMSE on available histories (mm)', ylim=(0,.33))
    for axis in axes:
        axis.set_xticks(x, labels, rotation=24, ha='right'); axis.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('8 reused development histories; 2048 final draws per replica; additional adaptation cost')
    fig.text(.5, .02, 'ESS / replica agreement do not certify accuracy. RMSE labels show availability; low-noise prior QMC uses only 2/4 histories.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94)); fig.savefig(root/'comparison.png', dpi=170); plt.close(fig)
    print(json.dumps({k: v for k, v in diagnostics.items() if k not in ('response_cases', 'notes')}, indent=2))
    print(json.dumps(cases, indent=2))


if __name__ == '__main__':
    main()
