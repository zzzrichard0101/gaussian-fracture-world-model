"""Read-only integration stability and unchanged-policy prediction analysis."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from history_refinement import digest,write_json
from report_observation_budget import paired_contrast


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--coordinate-run',type=Path);args=parser.parse_args()
    project=Path(os.environ['WM_PROJECT_ROOT']);root=args.run
    names=('report.json','forecasts.json','refinement-checks.json','adaptive-checks.json','evaluation-rows.json','inputs.json')
    hashes={n:digest(root/n) for n in names}
    report=json.loads((root/'report.json').read_text());settings=report['settings']
    rows=json.loads((root/'evaluation-rows.json').read_text());bundle=json.loads((root/'inputs.json').read_text())
    forecasts=json.loads((root/'forecasts.json').read_text());checks=json.loads((root/'refinement-checks.json').read_text())
    audit=json.loads((root/'adaptive-checks.json').read_text());primary=settings['primary_level']
    own={(r['episode_id'],r['method']):r for r in forecasts}
    baseline={(r['episode_id'],r['grid_side']):r for r in bundle['baseline_forecasts']}
    old_changes=[];old_to_new=[]
    for episode in bundle['episodes']:
        coarse,fine=baseline[(episode['id'],161)],baseline[(episode['id'],321)]
        old_changes.append(max(abs(a['mean_gap_m']-b['mean_gap_m']) for a,b in zip(coarse['futures'],fine['futures'])))
        new=own[(episode['id'],primary)]
        old_to_new.append({'episode_id':episode['id'],'sigma_m':episode['sigma_m'],
            'noise_condition':episode['noise_condition'],'policy':episode['policy'],'budget':episode['budget'],
            'mean_squared_grid321_to_guided_change_mm2':float(np.mean([((a['mean_gap_m']-b['mean_gap_m'])*1000)**2
                for a,b in zip(fine['futures'],new['futures'])]))})
    plan=json.loads((project/'config/observation-budget-analysis.json').read_text());contrasts=[]
    for noise in [n['name'] for n in bundle['source_settings']['noise_conditions']]:
        for sigma in bundle['source_settings']['gap_noise_sigma_m']:
            subset=[r for r in rows if r['method']==primary and r['noise_condition']==noise and r['sigma_m']==sigma]
            for policy in ('weak_repeat','fixed_ramp','rest_repeat'):
                left={r['context_id']:r for r in subset if r['policy']=='adaptive_verified' and r['budget']==8}
                right={r['context_id']:r for r in subset if r['policy']==policy and r['budget']==8}
                contrasts.append({'kind':'adaptive_vs_baseline','noise_condition':noise,'sigma_m':sigma,'budget':8,
                    'right_policy':policy,**paired_contrast(left,right,plan)})
            for policy in bundle['source_settings']['policies']:
                left={r['context_id']:r for r in subset if r['policy']==policy and r['budget']==8}
                right={r['context_id']:r for r in subset if r['policy']==policy and r['budget']==2}
                contrasts.append({'kind':'budget_8_vs_2','noise_condition':noise,'sigma_m':sigma,'policy':policy,
                    **paired_contrast(left,right,plan)})
    stability=[]
    for target in [level['name'] for level in settings['levels'][1:]]:
        chosen=[r for r in checks if r['to']==target]
        stability.append({'to':target,'checkpoints':len(chosen),'failures':sum(not r['empirical_tolerance_pass'] for r in chosen),
            'maximum_mean_change_mm':max(r['maximum_mean_change_m'] for r in chosen)*1000,
            'p95_mean_change_mm':float(np.quantile([r['maximum_mean_change_m']*1000 for r in chosen],.95)),
            'maximum_event_mass_change':max(r['maximum_event_mass_change'] for r in chosen),
            'maximum_log_integral_change':max(r['log_integral_change'] for r in chosen)})
    coordinate=None
    if args.coordinate_run:
        coordinate_path=args.coordinate_run/'report.json';coordinate=json.loads(coordinate_path.read_text())
        assert coordinate['source_run_id']==report['source_run_id']
    summary={'stability':stability,'maximum_original_grid_change_mm':max(old_changes)*1000,
        'coordinate_only_failures':coordinate['refinement_failures'] if coordinate else None,
        'coordinate_only_maximum_mean_change_mm':coordinate['maximum_last_mean_change_m']*1000 if coordinate else None,
        'adaptive_audit_failed_checks':sum(not r['empirical_tolerance_pass'] for r in audit),
        'adaptive_audit_maximum_mean_difference_mm':max(r['maximum_mean_difference_m'] for r in audit)*1000,
        'adaptive_audit_maximum_event_mass_difference':max(r['maximum_event_mass_difference'] for r in audit),
        'adaptive_audit_maximum_log_integral_change':max(r['log_integral_change'] for r in audit),
        'paired_development_contrasts':contrasts,'grid321_to_primary_predictions':old_to_new,
        'analysis_source_sha256':digest(Path(__file__)),'bootstrap_plan':plan,
        'raw_forecasts_and_evaluation_unchanged':True,'no_formal_integration_bound':True}
    write_json(root/'analysis-summary.json',summary)
    lines=['# 관측 반응 좌표와 관측식을 이용한 적분', '',
        '| 잡음 | nominal sigma (mm) | 정책 | grid321 RMSE (mm) | guided RMSE (mm) |',
        '| --- | ---: | --- | ---: | ---: |']
    for noise in [n['name'] for n in bundle['source_settings']['noise_conditions']]:
        for sigma in bundle['source_settings']['gap_noise_sigma_m']:
            for policy in bundle['source_settings']['policies']:
                select=lambda method:next(g for g in report['groups'] if (g['noise_condition'],g['sigma_m'],g['policy'],g['budget'],g['method'])==(noise,sigma,policy,8,method))
                a,b=select('grid_321'),select(primary)
                lines.append(f"| {noise} | {sigma*1000:.3f} | {policy} | {a['gap_rmse_mm']:.6f} | {b['gap_rmse_mm']:.6f} |")
    (root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.8),constrained_layout=True)
    labels=['Cartesian\n161 to 321'];values=[max(old_changes)*1000]
    if coordinate:
        labels+=['Response\n32 to 64'];values+=[coordinate['maximum_last_mean_change_m']*1000]
    labels+=['Guided\n32 to 64'];values+=[stability[-1]['maximum_mean_change_mm']]
    axes[0].bar(labels,values,color=['#8797a8','#d39a3c','#2a8b76'][-len(values):]);axes[0].set_yscale('log')
    axes[0].axhline(settings['mean_change_tolerance_m']*1000,color='black',ls='--',lw=1,label='Mean-change tolerance')
    axes[0].set(ylabel='Maximum forecast-mean change (mm)',title='Refinement differences\n(not certified error bounds)');axes[0].legend(fontsize=8)
    policies=bundle['source_settings']['policies'];x=np.arange(len(policies))
    for axis,noise in zip(axes[1:],['iid_bounded','correlated_bounded']):
        for shift,method,label in [(-.18,'grid_321','Cartesian 321'),(.18,primary,'Guided 64')]:
            values=[next(g['gap_rmse_mm'] for g in report['groups'] if (g['noise_condition'],g['sigma_m'],g['policy'],g['budget'],g['method'])==(noise,.000001,p,8,method)) for p in policies]
            axis.bar(x+shift,values,width=.35,label=label)
        axis.set_xticks(x,['Rest','Weak','Fixed','Adaptive']);axis.set_ylim(0,.25)
        title='Independent noise' if noise=='iid_bounded' else 'Correlated noise'
        axis.set(ylabel='Future gap RMSE (mm)',title=f'{title}\nNominal sigma 0.001 mm; 8 extra observations');axis.legend(fontsize=8)
    for axis in axes:axis.grid(axis='y',alpha=.2)
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    assert hashes=={n:digest(root/n) for n in names}
    out=project/'artifacts/guided-quadrature';out.mkdir(parents=True,exist_ok=True)
    for name in ('analysis-summary.json','comparison.png','summary.md'):shutil.copy2(root/name,out/name)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('paired_development_contrasts','grid321_to_primary_predictions','bootstrap_plan')},indent=2))


if __name__=='__main__':main()
