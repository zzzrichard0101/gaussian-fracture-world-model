"""Read-only paired comparisons against both phases of a clocked baseline."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from history_refinement import digest,write_json
from report_observation_budget import paired_contrast


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    root=args.run;project=Path(os.environ['WM_PROJECT_ROOT'])
    report=json.loads((root/'report.json').read_text());settings=report['settings'];comparison=report['comparison']
    names=('report.json','decisions.json','public-episodes.json','forecasts.json','evaluation-rows.json','control-replay.json')
    before={n:digest(root/n) for n in names}
    rows=json.loads((root/'evaluation-rows.json').read_text());primary=settings['primary_level']
    contrasts=[];phase_contrasts=[];costs=[]
    for noise in [n['name'] for n in settings['noise_conditions']]:
        for sigma in settings['gap_noise_sigma_m']:
            subset=[r for r in rows if r['noise_condition']==noise and r['sigma_m']==sigma and r['method']==primary]
            for budget in settings['budgets']:
                target={r['context_id']:r for r in subset if r['policy']=='future_task' and r['budget']==budget}
                for baseline in ('alternating_zero_first','alternating_probe_first'):
                    simple={r['context_id']:r for r in subset if r['policy']==baseline and r['budget']==budget}
                    contrast=paired_contrast(target,simple,comparison)
                    contrasts.append({'noise_condition':noise,'sigma_m':sigma,'budget':budget,'baseline':baseline,**contrast})
                    ratios=[target[k]['policy_seconds']/simple[k]['policy_seconds'] for k in sorted(target)]
                    costs.append({'noise_condition':noise,'sigma_m':sigma,'budget':budget,'baseline':baseline,
                        'mean_task_ms':float(np.mean([r['policy_seconds'] for r in target.values()])*1000),
                        'mean_baseline_ms':float(np.mean([r['policy_seconds'] for r in simple.values()])*1000),
                        'median_paired_time_ratio_task_over_baseline':float(np.median(ratios)),
                        'scope':'Contemporaneous rotated-order single-run timing; no independent hardware replication.'})
                a={r['context_id']:r for r in subset if r['policy']=='alternating_zero_first' and r['budget']==budget}
                b={r['context_id']:r for r in subset if r['policy']=='alternating_probe_first' and r['budget']==budget}
                phase_contrasts.append({'noise_condition':noise,'sigma_m':sigma,'budget':budget,**paired_contrast(a,b,comparison)})
    checks=json.loads((root/'refinement-checks.json').read_text())
    summary={'paired_task_vs_baseline':contrasts,'zero_first_minus_probe_first':phase_contrasts,'timing':costs,
        'forecast_refinement_failures':sum(not c['empirical_tolerance_pass'] for c in checks),
        'maximum_forecast_refinement_mean_change_m':max(c['maximum_mean_change_m'] for c in checks),
        'all_original_control_histories_and_losses_reproduced':True,
        'primary_comparison_predeclared_zero_first':True,'no_noninferiority_test':True,
        'analysis_source_sha256':digest(Path(__file__)),'original_raw_results_unchanged':True}
    write_json(root/'analysis-summary.json',summary)
    lines=['# 무하중과 검증 입력을 교대하는 기준선','',
        '이전과 같은 12개 개발 상태·잡음을 재사용하는 후속 기준선 점검이다. 각 행은 12상태×2편향=24조건이며 10개 설계 cluster를 공유한다.','',
        '| 잡음 | nominal sigma (mm) | 추가 관측 | 정책 | 미래 RMSE (mm) | 최대 추가 D | 미충족 예산 | 평균 누적 정책 시간 (ms) |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for noise in [n['name'] for n in settings['noise_conditions']]:
        for sigma in settings['gap_noise_sigma_m']:
            for budget in settings['budgets']:
                for policy in settings['policies']:
                    g=next(g for g in report['groups'] if (g['noise_condition'],g['sigma_m'],g['budget'],g['policy'],g['method'])==(noise,sigma,budget,policy,primary))
                    lines.append(f"| {noise} | {sigma*1000:.3f} | {budget} | {policy} | {g['gap_rmse_mm']:.6f} | {g['maximum_extra_damage']:.6f} | {g['unfilled_budgets']} | {g['mean_policy_ms']:.1f} |")
    (root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    labels={'future_task':'Future task score','alternating_zero_first':'Zero then verified probe','alternating_probe_first':'Verified probe then zero'}
    upper=1.12*max(g['gap_rmse_mm'] for g in report['groups'] if g['method']==primary)
    for ni,noise in enumerate([n['name'] for n in settings['noise_conditions']]):
        for si,sigma in enumerate(settings['gap_noise_sigma_m']):
            axis=axes[ni,si]
            for policy in settings['policies']:
                selected=[next(g for g in report['groups'] if (g['noise_condition'],g['sigma_m'],g['budget'],g['policy'],g['method'])==(noise,sigma,b,policy,primary)) for b in settings['budgets']]
                axis.plot(settings['budgets'],[g['gap_rmse_mm'] for g in selected],marker='o',label=labels[policy])
            title='Independent noise' if noise=='iid_bounded' else 'Correlated noise'
            axis.set(title=f'{title}; nominal sigma {sigma*1000:.3f} mm',xlabel='Additional observations',ylabel='Future gap RMSE (mm)')
            axis.set_xticks(settings['budgets']);axis.set_ylim(0,upper);axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    assert before=={n:digest(root/n) for n in names}
    out=project/'artifacts/alternating-frontier';out.mkdir(parents=True,exist_ok=True)
    for name in ('analysis-summary.json','comparison.png','summary.md'):shutil.copy2(root/name,out/name)
    selected=[c for c in contrasts if c['sigma_m']==comparison['primary_sigma_m'] and c['budget']==comparison['primary_budget'] and c['baseline']==comparison['primary_policy']]
    print(json.dumps({'primary_comparisons':selected,'forecast_refinement_failures':summary['forecast_refinement_failures'],
        'original_raw_results_unchanged':True},indent=2))


if __name__=='__main__':main()
