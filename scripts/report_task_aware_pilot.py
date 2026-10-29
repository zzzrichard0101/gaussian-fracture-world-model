"""Read-only paired utility analysis and tighter public acquisition audits."""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from history_refinement import digest,write_json
from report_observation_budget import paired_contrast
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.guided_quadrature import GuidedResponseQuadrature
from wmfracture.task_design import PredictiveMixture


def tight_scores(root,settings,guard,law,fixture,plan):
    decisions=json.loads((root/'decisions.json').read_text());reference_settings=deepcopy(settings)
    reference_settings['acquisition_levels']=plan['tight_acquisition_levels'];checks=[]
    for decision in decisions:
        if decision['policy']!='future_task':continue
        parts=decision['context_id'].split('-');record=decision['plan']
        if (int(parts[1]) not in plan['tight_audit_case_indices'] or int(parts[4])!=plan['tight_audit_bias_index']
                or decision['round'] not in plan['tight_audit_turns'] or not record['scores'] or record['acquisition_abstained']):continue
        prefix=decision['public_prefix'];model=GuidedResponseQuadrature(law,fixture,guard['parameter_bounds'],
            prefix['actions_m'],prefix['observed_gaps_m'],decision['sigma_m'],guard['bias_sigma_m'])
        reference=model.integrate(settings['future_drives_m'],**settings['belief_levels'][-1])
        mixture=PredictiveMixture(model,reference,settings);scores=[]
        for item in record['scores']:
            value,detail=mixture.task_score(item['action_m'],reference_settings)
            scores.append({'action_m':item['action_m'],'score':value,'stable':detail['stable']})
        stable=all(s['stable'] for s in scores)
        if stable:
            maximum=max(s['score'] for s in scores)
            chosen=next(s['score'] for s in scores if s['action_m']==decision['action_m'])
            regret=max(0.,maximum-chosen)
            tolerance=max(record.get('empirical_rank_tolerance_mm2',0.),settings['minimum_gain_mm2'])
            difference=max(abs(a['score']-b['score']) for a,b in zip(scores,record['scores']))
        else:regret=None;tolerance=None;difference=None
        checks.append({'trajectory_id':decision['trajectory_id'],'round':decision['round'],
            'all_reference_scores_stable':stable,'reference_regret_mm2':regret,
            'maximum_score_difference_mm2':difference,'decision_tolerance_mm2':tolerance,
            'selected_action_consistent_with_reference':bool(stable and regret<=tolerance),
            'reference_scores':scores})
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    root=args.run;project=Path(os.environ['WM_PROJECT_ROOT'])
    report=json.loads((root/'report.json').read_text());settings=report['settings']
    guard=json.loads((root/'guard-config.json').read_text())
    bundle=json.loads((root/'inputs.json').read_text());law=InterfaceLaw(**bundle['law']);fixture=Fixture(**settings['fixture'])
    plan_path=root/'analysis-plan.json'
    if not plan_path.exists():shutil.copy2(project/'config/task-aware-analysis.json',plan_path)
    plan=json.loads(plan_path.read_text())
    names=('report.json','decisions.json','public-episodes.json','private-cases.json','forecasts.json','evaluation-rows.json')
    before={name:digest(root/name) for name in names}
    rows=json.loads((root/'evaluation-rows.json').read_text());primary=settings['primary_level'];contrasts=[]
    for noise in [n['name'] for n in settings['noise_conditions']]:
        for sigma in settings['gap_noise_sigma_m']:
            subset=[r for r in rows if r['method']==primary and r['sigma_m']==sigma and r['noise_condition']==noise]
            for budget in settings['budgets']:
                for baseline in [p for p in settings['policies'] if p!='future_task']:
                    left={r['context_id']:r for r in subset if r['policy']=='future_task' and r['budget']==budget}
                    right={r['context_id']:r for r in subset if r['policy']==baseline and r['budget']==budget}
                    contrasts.append({'kind':'task_vs_baseline','noise_condition':noise,'sigma_m':sigma,'budget':budget,
                        'baseline':baseline,**paired_contrast(left,right,plan)})
    checks=tight_scores(root,settings,guard,law,fixture,plan)
    write_json(root/'tight-score-audits.json',checks)
    decisions=json.loads((root/'decisions.json').read_text());frontier=[]
    for policy in ('observation_disagreement','future_task'):
        for sigma in settings['gap_noise_sigma_m']:
            chosen=[d for d in decisions if d['policy']==policy and d['sigma_m']==sigma]
            first=[d['plan']['frontier_attempts'][0] for d in chosen]
            attempts=[a for d in chosen for a in d['plan']['frontier_attempts']]
            frontier.append({'policy':policy,'sigma_m':sigma,'rounds':len(chosen),
                'first_status':{kind:sum(a['decision']==kind for a in first) for kind in ('allow','reject','abstain','inconsistent')},
                'all_query_status':{kind:sum(a['decision']==kind for a in attempts) for kind in ('allow','reject','abstain','inconsistent')},
                'mean_ceiling_to_hint_ratio':float(np.mean([d['plan']['ceiling_m']/d['plan']['frontier_proposal_m']
                    for d in chosen if d['plan']['frontier_proposal_m'] and 'ceiling_m' in d['plan']]))})
    adaptive={(d['context_id'],d['round'],d['policy']):d for d in decisions if d['policy'] in ('observation_disagreement','future_task')}
    pairs=[(d,adaptive[(d['context_id'],d['round'],'future_task')]) for d in decisions if d['policy']=='observation_disagreement']
    same_prefix=[(a,b) for a,b in pairs if a['public_prefix']==b['public_prefix']]
    summary={'paired_contrasts':contrasts,'tight_audits':len(checks),
        'tight_audit_selection_changes':sum(not r['selected_action_consistent_with_reference'] for r in checks),
        'tight_audit_reference_instabilities':sum(not r['all_reference_scores_stable'] for r in checks),
        'maximum_tight_reference_regret_mm2':max((r['reference_regret_mm2'] for r in checks if r['reference_regret_mm2'] is not None),default=None),
        'analysis_plan':plan,'analysis_source_sha256':digest(Path(__file__)),
        'frontier_diagnostics':frontier,'paired_adaptive_rounds':len(pairs),
        'identical_public_prefix_pairs':len(same_prefix),
        'different_actions_at_identical_prefix':sum(a['action_m']!=b['action_m'] for a,b in same_prefix),
        'same_action_across_all_paired_rounds':sum(a['action_m']==b['action_m'] for a,b in pairs),
        'raw_decisions_observations_forecasts_unchanged':True}
    write_json(root/'analysis-summary.json',summary)
    lines=['#  목표 미래의 예측 위험을 줄이는 입력 선택','',
        '새로운 12개 합성 상태·10개 생성 cluster의 작은 개발 pilot이다. 각 칸은 12상태×2편향=24조건이며 독립 시편 24개가 아니다.','',
        '| 잡음 | nominal sigma (mm) | 추가 관측 | 정책 | 미래 RMSE (mm) | 한도 위반 | 미충족 예산 | 점수 계산 보류 | 평균 누적 정책 시간 (ms) |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for noise in [n['name'] for n in settings['noise_conditions']]:
        for sigma in settings['gap_noise_sigma_m']:
            for budget in settings['budgets']:
                for policy in settings['policies']:
                    g=next(g for g in report['groups'] if (g['noise_condition'],g['sigma_m'],g['budget'],g['policy'],g['method'])==(noise,sigma,budget,policy,primary))
                    lines.append(f"| {noise} | {sigma*1000:.3f} | {budget} | {policy} | {g['gap_rmse_mm']:.5f} | {g['cap_violations']} | {g['unfilled_budgets']} | {g['acquisition_abstentions']} | {g['mean_policy_ms']:.1f} |")
    (root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    labels={'rest_repeat':'Rest','weak_repeat':'Weak','fixed_ramp':'Fixed + proof','calibration_alternating':'Zero/weak alternation',
        'observation_disagreement':'Observation proxy','future_task':'Future task gain'}
    upper=1.12*max(g['gap_rmse_mm'] for g in report['groups'] if g['method']==primary)
    for ni,noise in enumerate([n['name'] for n in settings['noise_conditions']]):
        for si,sigma in enumerate(settings['gap_noise_sigma_m']):
            axis=axes[ni,si]
            for policy in settings['policies']:
                chosen=[next(g for g in report['groups'] if (g['noise_condition'],g['sigma_m'],g['budget'],g['policy'],g['method'])==(noise,sigma,b,policy,primary)) for b in settings['budgets']]
                axis.plot(settings['budgets'],[g['gap_rmse_mm'] for g in chosen],marker='o',label=labels[policy])
            title='Independent noise' if noise=='iid_bounded' else 'Correlated noise'
            axis.set(title=f'{title}; nominal sigma {sigma*1000:.3f} mm',xlabel='Additional observations',ylabel='Future gap RMSE (mm)')
            axis.set_xticks(settings['budgets']);axis.set_ylim(0,upper);axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    assert before=={name:digest(root/name) for name in names}
    out=project/'artifacts/task-aware-pilot';out.mkdir(parents=True,exist_ok=True)
    for name in ('analysis-plan.json','analysis-summary.json','tight-score-audits.json','summary.md','comparison.png'):
        shutil.copy2(root/name,out/name)
    primary_comparisons=[c for c in contrasts if c['sigma_m']==plan['primary_sigma_m'] and c['budget']==plan['primary_budget'] and c['baseline']=='fixed_ramp']
    print(json.dumps({'primary_comparisons':primary_comparisons,**{k:v for k,v in summary.items()
        if k not in ('paired_contrasts','analysis_plan','frontier_diagnostics')}},indent=2))


if __name__=='__main__':main()
