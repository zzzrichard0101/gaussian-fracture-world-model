"""Read-only timing analysis and scientific figure for paired trace replay."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from history_refinement import digest, write_json


def main():
    project=Path(os.environ['WM_PROJECT_ROOT']);out=project/'artifacts/risk-reuse'
    report=json.loads((out/'report.json').read_text());root=Path(report['run_directory'])
    hashes={name:digest(root/name) for name in ('report.json','rows.json','inputs.json')}
    rows=json.loads((root/'rows.json').read_text());groups=report['groups']
    repeats=[]
    for repeat in range(report['repetitions']):
        totals={mode:sum(r['policy_seconds'] for r in rows if r['repeat']==repeat and r['mode']==mode)
            for mode in ('cold','reuse')}
        repeats.append({'repeat':repeat,'cold_seconds':totals['cold'],'reuse_seconds':totals['reuse'],
            'speedup_ratio':totals['cold']/totals['reuse']})
    pairs=[]
    for repeat in range(report['repetitions']):
        for episode in {r['episode_id'] for r in rows}:
            pair={mode:sum(r['policy_seconds'] for r in rows if r['repeat']==repeat and r['episode_id']==episode and r['mode']==mode)
                for mode in ('cold','reuse')}
            pairs.append(pair['cold']/pair['reuse'])
    summary={'repeat_timings':repeats,'median_paired_episode_speedup':float(np.median(pairs)),
        'observation_constraint_reduction_fraction':1-sum(g[k] for g in groups[1:] for k in
            ('search_box_constraints','search_point_constraints','checker_box_constraints','checker_point_constraints'))/
            sum(groups[0][k] for k in ('search_box_constraints','search_point_constraints','checker_box_constraints','checker_point_constraints')),
        'scope':'Repeated timings of reused development traces, not independent predictive or hardware replications.'}
    write_json(out/'analysis-summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2));colors=['#677b8d','#258a74']
    axes[0].bar(['Cold','Reuse'],[g['mean_two_step_ms'] for g in groups],color=colors)
    axes[0].set_ylabel('Mean two-step policy time (ms)')
    for i,g in enumerate(groups):axes[0].text(i,g['mean_two_step_ms']+3,f"{g['mean_two_step_ms']:.1f}",ha='center')
    constraints=[sum(g[k] for k in ('search_box_constraints','search_point_constraints','checker_box_constraints','checker_point_constraints')) for g in groups]
    axes[1].bar(['Cold','Reuse'],np.asarray(constraints)/1e6,color=colors)
    axes[1].set_ylabel('Observation constraint evaluations (millions)')
    for i,c in enumerate(constraints):axes[1].text(i,c/1e6+.015,f'{c:,}',ha='center',fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,ax.get_ylim()[1]*1.12)
    fig.suptitle('Same actions, decisions and selected proof trees | public scalar traces')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    assert hashes=={name:digest(root/name) for name in hashes}
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
