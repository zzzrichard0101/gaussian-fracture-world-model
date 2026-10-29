"""Read-only budget, paired cluster sensitivity and forecast-grid diagnostics."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from history_refinement import digest,write_json


def paired_contrast(left,right,plan):
    """RMSE(left)-RMSE(right); negative is lower error for left."""
    assert set(left)==set(right)
    keys=sorted(left)
    assert all(left[k]['cluster']==right[k]['cluster'] for k in keys)
    clusters=sorted({left[k]['cluster'] for k in keys})
    sums=np.asarray([[sum(left[k]['future_mean_squared_error_mm2'] for k in keys if left[k]['cluster']==c),
        sum(right[k]['future_mean_squared_error_mm2'] for k in keys if left[k]['cluster']==c),
        sum(left[k]['cluster']==c for k in keys)] for c in clusters])
    rng=np.random.default_rng(plan['bootstrap_seed'])
    selections=rng.integers(0,len(clusters),size=(plan['bootstrap_repetitions'],len(clusters)))
    samples=sums[selections].sum(axis=1)
    delta=np.sqrt(samples[:,0]/samples[:,2])-np.sqrt(samples[:,1]/samples[:,2])
    l=float(np.sqrt(np.mean([left[k]['future_mean_squared_error_mm2'] for k in keys])))
    r=float(np.sqrt(np.mean([right[k]['future_mean_squared_error_mm2'] for k in keys])))
    return {'left_rmse_mm':l,'right_rmse_mm':r,'delta_rmse_mm':l-r,
        'relative_reduction_fraction':1-l/r if r else None,
        'cluster_bootstrap_95_percentile_delta_mm':np.quantile(delta,[.025,.975]).tolist(),
        'paired_contexts':len(keys),'designed_clusters':len(clusters)}


def analyze(root,plan):
    report=json.loads((root/'report.json').read_text());settings=report['settings']
    rows=json.loads((root/'evaluation-rows.json').read_text());primary=plan['primary_grid_side']
    contrasts=[]
    for condition in settings['noise_conditions']:
        noise=condition['name']
        for sigma in settings['gap_noise_sigma_m']:
            subset=[r for r in rows if r['grid_side']==primary and r['sigma_m']==sigma and r['noise_condition']==noise]
            for budget in settings['budgets']:
                for baseline in ('rest_repeat','weak_repeat','fixed_ramp'):
                    left={r['context_id']:r for r in subset if r['policy']=='adaptive_verified' and r['budget']==budget}
                    right={r['context_id']:r for r in subset if r['policy']==baseline and r['budget']==budget}
                    contrasts.append({'kind':'adaptive_vs_baseline','noise_condition':noise,'sigma_m':sigma,
                        'budget':budget,'left_policy':'adaptive_verified','right_policy':baseline,
                        **paired_contrast(left,right,plan)})
            for policy in settings['policies']:
                left={r['context_id']:r for r in subset if r['policy']==policy and r['budget']==8}
                right={r['context_id']:r for r in subset if r['policy']==policy and r['budget']==2}
                contrasts.append({'kind':'budget_8_vs_2','noise_condition':noise,'sigma_m':sigma,
                    'policy':policy,**paired_contrast(left,right,plan)})
    predictions=json.loads((root/'forecasts.json').read_text())
    predictions={(r['episode_id'],r['grid_side']):r for r in predictions}
    grid_rows=[]
    for row in rows:
        if row['grid_side']!=primary:continue
        fine=predictions[(row['episode_id'],primary)];coarse=predictions[(row['episode_id'],161)]
        shift=float(np.mean([((a['mean_gap_m']-b['mean_gap_m'])*1000)**2 for a,b in zip(fine['futures'],coarse['futures'])]))
        grid_rows.append({**{k:row[k] for k in ('episode_id','noise_condition','sigma_m','budget','policy')},
            'grid_prediction_shift_mse_mm2':shift})
    grids=[]
    for key in sorted({(r['noise_condition'],r['sigma_m'],r['budget'],r['policy']) for r in grid_rows}):
        chosen=[r for r in grid_rows if (r['noise_condition'],r['sigma_m'],r['budget'],r['policy'])==key]
        fine=next(g for g in report['groups'] if (g['noise_condition'],g['sigma_m'],g['budget'],g['policy'],g['grid_side'])==(*key,primary))
        coarse=next(g for g in report['groups'] if (g['noise_condition'],g['sigma_m'],g['budget'],g['policy'],g['grid_side'])==(*key,161))
        grids.append({**dict(zip(('noise_condition','sigma_m','budget','policy'),key)),
            'mean_prediction_grid_shift_rmse_mm':float(np.sqrt(np.mean([r['grid_prediction_shift_mse_mm2'] for r in chosen]))),
            'coarse_gap_rmse_mm':coarse['gap_rmse_mm'],'fine_gap_rmse_mm':fine['gap_rmse_mm']})
    return report,rows,{'plan':plan,'paired_contrasts':contrasts,'forecast_grid_diagnostics':grids,
        'raw_results_unchanged':True,'analysis_is_descriptive_development_comparison':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    root=args.run;project=Path(os.environ['WM_PROJECT_ROOT'])
    plan_path=root/'analysis-plan.json'
    if not plan_path.exists():
        shutil.copy2(project/'config/observation-budget-analysis.json',plan_path)
    plan=json.loads(plan_path.read_text())
    names=('report.json','forecasts.json','evaluation-rows.json','decisions.json','private-cases.json','public-episodes.json')
    hashes={n:digest(root/n) for n in names}
    report,rows,summary=analyze(root,plan);settings=report['settings'];primary=plan['primary_grid_side']
    summary['analysis_source_sha256']=digest(Path(__file__))
    summary['analysis_plan_sha256']=digest(plan_path)
    write_json(root/'analysis-summary.json',summary)
    lines=['# 관측 횟수와 입력 선택 비교','',
        '동일한 24개 개발 상태·20개 생성 cluster를 사용한 합성 준정적 스칼라 실험이다. 아래 각 행은 24상태×2편향=48조건이며 독립 시편 48개가 아니다. 2·4·8회는 같은 예산 미인지 greedy 실행의 중첩 prefix다.','',
        '모든 정책에 동일한 연속 영역 검증기를 적용하고 최초 zero/weak 관측 이후 누적 추가 D 한도 0.02를 유지했다. 기준 이력은 관측이 추가되어도 초기화하지 않는다. 새로운 보류 상태 시험이나 새 알고리즘 신규성의 증거는 아니다.','',
        '| 잡음 | nominal sigma (mm) | 추가 관측 | 정책 | 미래 RMSE (mm) | 위반/48 | 미충족 예산/48 | 최대 추가 D | 누적 정책 시간 (ms) |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for noise in [c['name'] for c in settings['noise_conditions']]:
        for sigma in settings['gap_noise_sigma_m']:
            for budget in settings['budgets']:
                for policy in settings['policies']:
                    g=next(g for g in report['groups'] if (g['noise_condition'],g['sigma_m'],g['budget'],g['policy'],g['grid_side'])==(noise,sigma,budget,policy,primary))
                    lines.append(f"| {noise} | {sigma*1000:.3f} | {budget} | {policy} | {g['gap_rmse_mm']:.4f} | {g['cap_violations']} | {g['unfilled_budgets']} | {g['maximum_extra_damage']:.5f} | {g['mean_policy_ms']:.2f} |")
    lines+=['','## 사전 지정한 주 비교: 0.020mm, 추가 관측 8회, adaptive − weak','',
        '| 잡음 | adaptive RMSE | weak RMSE | 차이 (mm) | cluster 재표집 95% 범위 (mm) |',
        '| --- | ---: | ---: | ---: | --- |']
    primary_contrasts=[c for c in summary['paired_contrasts'] if c['kind']=='adaptive_vs_baseline'
        and c['sigma_m']==plan['primary_noise_sigma_m'] and c['budget']==plan['primary_budget'] and c['right_policy']=='weak_repeat']
    for c in primary_contrasts:
        lo,hi=c['cluster_bootstrap_95_percentile_delta_mm']
        lines.append(f"| {c['noise_condition']} | {c['left_rmse_mm']:.4f} | {c['right_rmse_mm']:.4f} | {c['delta_rmse_mm']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines+=['','재표집은 matched 쌍과 두 편향 조건을 cluster째 묶는다. 설계한 20개 개발 cluster의 서술적 민감도이며 일반 모집단 신뢰구간이나 최종 독립 검증으로 해석하지 않는다. 음수 차이는 adaptive의 낮은 오차를 뜻한다.','',
        '## 관측·추론·검증의 조건','',
        '- iid와 상관 잡음은 같은 잘린 정규 주변분포를 가지며 ±3 nominal sigma 안에 있다. 상관 조건은 정상 Gaussian AR(1) latent rho=0.8을 Gaussian copula로 변환한다. 변환 후 Pearson 상관을 정확히 0.8이라고 주장하지 않는다.',
        '- 실제 편향은 한 실행 내에서 일정하고, 반복 관측으로 없어진다고 가정하지 않는다. 같은 상태/sigma/편향에서 정책은 동일한 잡음을 사용하며, 두 상관 조건도 같은 innovation을 공유한다.',
        '- 모든 정책의 가중치는 기존 iid Gaussian+shared bias 근사를 쓴다. 잘린 잡음에서 근사이며 상관 조건에서는 시간 공분산도 생략한다. 95% 구간은 명목 구간으로 empirical 포함률만 보고하고 보정된 posterior를 주장하지 않는다.',
        '- 연속 영역 위험 검증에는 오차의 독립성이 필요하지 않다. 참 물성과 이력이 사전 안에 있고 모든 오차가 범위 안에 있는지 별도 점검했다. 알려진 물리 법칙에 조건부이며 실물 안전 보장은 아니다.',
        '- 모든 실제 입력의 증거를 실행 전에 검사하고 평가에서 새 cold 검사기로 다시 확인했다. 중단되면 관측을 만들어 채우지 않고 실제 획득한 관측으로 예측한다.',
        '- 미래 입력은 모든 정책에 같지만, 관측용 입력이 물체 상태를 바꿀 수 있으므로 각 정책이 만든 최종 상태의 미래를 평가한다. 서로 다른 상태를 무시한 동일 정답 비교가 아니다.',
        '- 주 예측은 321² 후보를 사용하며 161² 결과도 함께 보관한다. 두 해상도의 일치는 연속 posterior 수렴 증명이 아니다.','',
        f"원본 실행: `{root.name}`. 순차 제어 {report['collection_seconds']:.2f}초, 공통 예측 {report['forecast_seconds']:.2f}초, 전체 {report['total_seconds']:.2f}초. 정책별 시간은 초기 관측 이후 해당 checkpoint까지의 제안·warm bank·모든 검증·증거 저장 합계이며 관측 생성과 공통 예측은 제외했다. 실행 순서는 고정되어 있어 엄밀한 속도 우위 비교는 아니다.",'']
    (root/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    common_upper=1.12*max(g['gap_rmse_mm'] for g in report['groups'] if g['grid_side']==primary)
    labels={'rest_repeat':'Rest repeats','weak_repeat':'Weak repeats','fixed_ramp':'Fixed ramp + filter','adaptive_verified':'Adaptive + filter'}
    for ni,noise in enumerate([c['name'] for c in settings['noise_conditions']]):
        for si,sigma in enumerate(settings['gap_noise_sigma_m']):
            ax=axes[ni,si]
            for policy in settings['policies']:
                chosen=[next(g for g in report['groups'] if (g['noise_condition'],g['sigma_m'],g['budget'],g['policy'],g['grid_side'])==(noise,sigma,b,policy,primary)) for b in settings['budgets']]
                ax.plot(settings['budgets'],[g['gap_rmse_mm'] for g in chosen],marker='o',label=labels[policy])
            ax.set(title=f'{noise}; nominal sigma={sigma*1000:.3f} mm',xlabel='Additional observations after zero/weak prefix',ylabel='Future gap RMSE (mm)')
            ax.set_xticks(settings['budgets']);ax.grid(alpha=.2);ax.legend(fontsize=8)
            ax.set_ylim(0,common_upper)
    fig.suptitle('Same cumulative damage cap 0.02 | 24 development states | primary grid 321 x 321')
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    assert hashes=={n:digest(root/n) for n in names}
    out=project/'artifacts/observation-budget';out.mkdir(parents=True,exist_ok=True)
    for name in ('analysis-summary.json','summary.md','comparison.png','analysis-plan.json'):shutil.copy2(root/name,out/name)
    print(json.dumps({'primary_contrasts':primary_contrasts,'raw_results_unchanged':True},indent=2))


if __name__=='__main__':main()
