"""Derived budget audits and scientific figure; frozen estimates unchanged."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spatial_observation_evaluate import write


def main():
    project = Path(__file__).resolve().parents[1]; root = project/'artifacts/spatial-budget'
    report = json.loads((root/'report.json').read_text())
    costs = json.loads((root/'costs.json').read_text())
    rows = json.loads((root/'evaluation-rows.json').read_text())
    maximum = max(report['config']['endpoint_budgets'])
    assert all(r['used_endpoint_cost'] <= r['budget'] and r['unused_endpoint_budget'] < r['full_path_endpoint_cost_per_parameter'] for r in costs)
    audits = []
    for method in report['config']['methods']:
        group = [r for r in costs if r['method'] == method and r['budget'] == maximum]
        audits.append({'method': method, 'maximum_budget_runs': len(group),
            'setup_endpoint_range': [min(r['setup_endpoint_cost'] for r in group), max(r['setup_endpoint_cost'] for r in group)],
            'setup_fraction_range': [min(r['setup_endpoint_cost']/r['budget'] for r in group), max(r['setup_endpoint_cost']/r['budget'] for r in group)],
            'unused_endpoint_range': [min(r['unused_endpoint_budget'] for r in group), max(r['unused_endpoint_budget'] for r in group)],
            'outside_draws_range': [min(r['outside_cube_draws'] for r in group), max(r['outside_cube_draws'] for r in group)],
            'wall_seconds_range': [min(r['measured_maximum_budget_seconds'] for r in group), max(r['measured_maximum_budget_seconds'] for r in group)]})
    response_rows = [r for r in rows if r['method'] in ('response_chart','hybrid_equal') and r['budget'] == maximum]
    paired = []
    for sigma in (1e-6, 2e-5):
        wide = {r['id']: r for r in rows if r['method'] == 'physical_mixture' and r['sigma_m'] == sigma and r['budget'] == maximum}
        hybrid = {r['id']: r for r in rows if r['method'] == 'hybrid_equal' and r['sigma_m'] == sigma and r['budget'] == maximum}
        common = [key for key in wide if wide[key]['available'] and hybrid[key]['available']]
        paired.append({'sigma_m': sigma, 'common_available_histories': len(common),
            'physical_rmse_mm_common': float(np.sqrt(np.mean([wide[key]['mse_mm2'] for key in common]))) if common else None,
            'hybrid_rmse_mm_common': float(np.sqrt(np.mean([hybrid[key]['mse_mm2'] for key in common]))) if common else None,
            'hybrid_lower_mse_histories': sum(hybrid[key]['mse_mm2'] < wide[key]['mse_mm2'] for key in common),
            'both_screened_histories': sum(wide[key]['screen_passed'] and hybrid[key]['screen_passed'] for key in common)})
    write(root/'diagnostics.json', {'cost_rows_checked': len(costs), 'all_costs_within_cap': True,
        'maximum_unused_fraction_all_levels': max(r['unused_endpoint_budget']/r['budget'] for r in costs),
        'cost_audits': audits, 'paired_primary': paired, 'response_and_hybrid_screen_rows': response_rows,
        'notes': ['Wall times belong to maximum-budget fresh-bank method runs only; nested lower-budget times were not measured.',
            'Two budget levels and two independent full-adaptation replicas provide an empirical screen, not a posterior error enclosure.',
            'The primary comparison is eight exposed development histories, with four physical states; no significance or holdout claim.']})
    write(root/'report-source-sha256.json', {'scripts/report_spatial_budget.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    labels = {'prior_qmc':'Prior QMC', 'physical_mixture':'Pilot physical mix',
        'response_chart':'Response chart', 'hybrid_equal':'Broad + response'}
    fig, axes = plt.subplots(2,2,figsize=(11,7.5))
    for col, sigma in enumerate((1e-6,2e-5)):
        for method in report['config']['methods']:
            selected = sorted([r for r in report['summaries'] if r['sigma_m'] == sigma and r['method'] == method], key=lambda r:r['budget'])
            x = [r['budget']/1e6 for r in selected]
            axes[0,col].plot(x,[r['median_ess'] for r in selected],'o-',label=labels[method])
            axes[1,col].plot(x,[r['rmse_mm_on_available'] for r in selected],'o-',label=labels[method])
            for pos, row in zip(x, selected):
                if row['available'] != row['histories']:
                    axes[1,col].annotate(f"{row['available']}/{row['histories']} available", (pos,row['rmse_mm_on_available']),
                        xytext=(6,8),textcoords='offset points',fontsize=8)
        axes[0,col].set(title=f'sigma={sigma*1000:g} mm; 4 reused histories',ylabel='Median ESS, primary replica',yscale='log')
        axes[0,col].axhline(40,color='black',linestyle='--',linewidth=.8)
        axes[1,col].set(ylabel='Gap RMSE (mm), available histories',ylim=(0,.16),xlabel='Total endpoint-parameter budget (millions)')
        for axis in axes[:,col]: axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Matched physics budgets include pilot, adaptation, preparation and future branches')
    fig.text(.5,.015,'Numerical screens do not certify accuracy. Availability is reported in the tables; equal physics budgets do not imply equal CPU time.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps({'cost_audits':audits,'paired_primary':paired,'response_and_hybrid_screen_rows':response_rows},indent=2))


if __name__ == '__main__':
    main()
