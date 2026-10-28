"""Read-only analysis/figures for an already frozen evaluation run."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from history_refinement import digest,write_json


def aggregate(rows):
    return {'episodes':len(rows),'violations':sum(r['cap_violation'] for r in rows),
        'bound_errors':sum(not r['actual_errors_within_bounds'] for r in rows),
        'gap_rmse_mm':float(np.sqrt(np.mean([r['future_mean_squared_error_mm2'] for r in rows]))),
        'policy_seconds':float(np.mean([r['policy_seconds'] for r in rows])),
        'maximum_extra_damage':max(r['extra_damage'] for r in rows),
        'query_abstentions':sum(r['verification_abstentions'] for r in rows),
        'stops':sum(r['stopped_without_execution'] for r in rows),
        'interval_score':float(np.mean([r['interval_score_95_mm'] for r in rows])),
        'brier':float(np.mean([r['separation_brier'] for r in rows]))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    root=args.run;report=json.loads((root/'report.json').read_text());settings=report['settings']
    rows=json.loads((root/'evaluation-rows.json').read_text());steps=json.loads((root/'decisions.json').read_text())
    original={n:digest(root/n) for n in ('report.json','evaluation-rows.json','private-cases.json','public-episodes.json','forecasts.json','decisions.json','posteriors.npz')}
    summaries=[]
    for noise in settings['noise_families']:
        for policy in settings['policies']:
            subset=[r for r in rows if r['noise_family']==noise and r['policy']==policy]
            summaries.append({'noise_family':noise,'policy':policy,**aggregate(subset)})
    lines=['# 새로운 합성 상태에서의 연속 위험 필터 평가','',
        '물성·준비 이력이 이전 26개 상태와 겹치지 않는 24개 상태를 새로 생성했다. uniform 8개, 손상 시작 근처 8개, 같은 약한 반응을 갖는 4쌍(8개)이다. 20개 생성 cluster를 공유하므로 모든 관측 조건을 독립 시편으로 세지 않는다.','',
        '설정·seed·코드를 고정한 뒤 2개 sigma×2개 실제 bias×2개 잡음 방식×6개 정책을 실행했다. 총 1,152개 실행이며 같은 정책 비교에는 동일한 관측 잡음을 사용했다. 같은 sigma/bias 조합도 같은 상태를 반복 사용하므로 독립 반복이 아니다. 알려진 동일 물리 법칙 안의 개발 평가다.','',
        '## 오차 경계를 만족하는 주 평가','',
        'bounded_normal은 Gaussian 제안을 ±3 nominal sigma에서 rejection sampling한다. 실제 측정 표준편차를 실물에서 측정한 값이 아니며, 잘린 분포의 표준편차는 nominal sigma와 정확히 같지 않다. 추론기의 Gaussian weighting은 모든 정책에 같은 근사로 사용했고 posterior 보정 정리를 주장하지 않는다.','',
        '| 정책 | 실제 한도 위반 / 96 | 미래 RMSE (mm) | 최대 추가 D | 평균 계획·검증 시간 (ms) | 중간 질의 보류 | 중단 실행 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for policy in settings['policies']:
        r=next(r for r in summaries if r['noise_family']=='bounded_normal' and r['policy']==policy)
        lines.append(f"| {policy} | {r['violations']} / {r['episodes']} | {r['gap_rmse_mm']:.4f} | {r['maximum_extra_damage']:.5f} | {r['policy_seconds']*1000:.2f} | {r['query_abstentions']} | {r['stops']} |")
    lines+=['','중간 질의 보류는 bisection 도중의 미확정 판정이다. 정책이 실제 행동을 하지 못해 중단한 실행 수와 다르다. 검증 필터는 판단 보류를 위험 증거로 바꾸지 않고, 마지막으로 검증된 입력만 실행한다.','',
        '## 정밀도를 나눈 대표 조건: bounded normal, 실제 bias +0.030mm','',
        '| nominal sigma (mm) | 정책 | 미래 RMSE (mm) | 한도 위반 / 24 | 평균 전체 정책 시간 (ms) |',
        '| --- | --- | ---: | ---: | ---: |']
    for sigma in settings['gap_noise_sigma_m']:
        for policy in ('rest_only','full_guard','continuous_guard','verified_filter'):
            r=next(g for g in report['groups'] if g['noise_family']=='bounded_normal' and g['sigma_m']==sigma
                and g['actual_bias_m']==.00003 and g['policy']==policy)
            lines.append(f"| {sigma*1000:.3f} | {policy} | {r['gap_rmse_mm']:.4f} | {r['cap_violations']} / 24 | {r['mean_policy_seconds']*1000:.2f} |")
    lines+=['','## 자르지 않은 Gaussian 대조군','',
        '| 정책 | 실제 한도 위반 / 96 | 실제 오차 경계 이탈 실행 | 미래 RMSE (mm) |',
        '| --- | ---: | ---: | ---: |']
    for policy in settings['policies']:
        r=next(r for r in summaries if r['noise_family']=='gaussian' and r['policy']==policy)
        lines.append(f"| {policy} | {r['violations']} / {r['episodes']} | {r['bound_errors']} | {r['gap_rmse_mm']:.4f} |")
    lines+=['','이번 Gaussian 표본에는 실제로 3 sigma 경계를 넘은 오차가 없었다. 따라서 잡음 경계가 깨진 상황에 대한 안전 검증을 한 결과가 아니다. 정책과 sigma/bias에 공유한 난수를 독립 표본으로 곱해 계산하지 않는다. 경계 밖 잡음을 자르지 않는 동작은 별도 테스트로 확인했다.','',
        '## 비용과 추론/평가 분리','',
        f"데이터 생성과 순차 제어는 {report['collection_seconds']:.2f}초, 공통 미래 예측은 {report['forecast_seconds']:.2f}초, 평가까지 {report['total_seconds']:.2f}초였다. CPU 구현이다. 정책 시간은 제안·warm bank·영역 검증·모든 중간 proof replay·실행 증거 저장을 포함한다. 관측 생성과 미래 예측은 이 정책 시간에서 제외하고 별도 집계했다.",'',
        'warm bank의 논리 root 조회, 영역 수, proof replay 시간도 개별 평가에 있다. 고정된 정책 실행 순서와 프로세스 캐시 영향을 포함한 한 번의 CPU wall 측정이며, 속도 우위를 검증한 반복 timing 시험은 아니다. full과 witness의 작은 시간 차이를 성능 차이로 주장하지 않는다.','',
        '예측은 실제 실행한 자기 정책의 마지막 상태에서 계산했고 private 정답을 읽기 전에 고정했다. 지연/중단도 실제 사용한 관측 수를 유지한다. 모든 실제 실행 allow proof를 행동 전에 재검사하고 평가 단계에서도 관측·입력과 대조했다. 중간 판정의 덮개는 즉시 검사했지만 저장하지 않고, 실제 실행한 입력의 전체 덮개만 원본 실행에 보관한다.','',
        '## 판단','',
        '1. 검증 없는 연속 입력의 실제 누락이 새 상태에서도 발생했다. 연속 영역 필터는 선언한 범위 안의 주 평가에서 한도를 지켰다.',
        '2. 0.001mm 가상 정밀도에서는 coarse/rest 대비 예측 오차가 줄었지만, 검증 없는 연속 입력보다 보수적이었다. 같은 안전 요구를 만족하지 않는 방법의 낮은 오차를 공정한 안전 성능 우위라고 부르지 않는다.',
        '3. 0.020mm 조건에서는 의미 있는 예측 이득을 확보했다고 보기 어렵다. 이 정도 오차에서도 유용한 관측을 얻는지는 여전히 해결되지 않았다.',
        '4. 검증 시간 부담이 크다. 현재 구현은 제안/지지점의 물리 계산을 일부 반복하고 모든 중간 증거도 재검사한다. 실행 증거만 검사하는 개선이나 영역 재사용은 다음 비교 대상으로 두되 성능을 미리 주장하지 않는다.',
        '5. 현재 결과는 같은 알려진 법칙의 scalar gap 입력이며 RGB·Gaussian splatting·실제 카메라·일반 3D/관성 운동의 결과가 아니다. 이 표준 proposal/backtracking/box verification 결합으로 높은 신규성이 확정되지 않는다.','']
    (root/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(root/'analysis-summary.json',{'aggregates':summaries,'analysis_source_sha256':digest(Path(__file__)),
        'executed_allow_proof_count':sum(s['executed_proof'] is not None for s in steps),
        'query_abstentions_total':sum(r['verification_abstentions'] for r in rows),
        'raw_results_unchanged':True})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    policies=('rest_only','full_guard','continuous_guard','verified_filter');labels=('rest','coarse full','continuous','verified')
    for index,sigma in enumerate(settings['gap_noise_sigma_m']):
        chosen=[next(g for g in report['groups'] if g['noise_family']=='bounded_normal' and g['sigma_m']==sigma
            and g['actual_bias_m']==.00003 and g['policy']==policy) for policy in policies]
        bars=axes[index].bar(labels,[r['gap_rmse_mm'] for r in chosen])
        for bar,row in zip(bars,chosen):
            axes[index].annotate(f"{row['cap_violations']}/24 violations",(bar.get_x()+bar.get_width()/2,bar.get_height()),
                xytext=(0,4),textcoords='offset points',ha='center',fontsize=7)
        axes[index].set_ylim(0,max(r['gap_rmse_mm'] for r in chosen)*1.2)
        axes[index].set(ylabel='Future hold-gap RMSE (mm)',title=f'Nominal sigma={sigma*1000:.3f} mm; true bias +0.030 mm')
    chosen=[next(g for g in summaries if g['noise_family']=='bounded_normal' and g['policy']==policy) for policy in policies]
    axes[2].bar(labels,[r['policy_seconds']*1000 for r in chosen]);axes[2].set(ylabel='Mean policy wall time (ms)',title='Includes verification, replay and proof storage')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelrotation=20)
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    assert original=={n:digest(root/n) for n in original}
    out=Path(os.environ['WM_PROJECT_ROOT'])/'artifacts/continuous-risk-evaluation'
    for name in ('summary.md','analysis-summary.json','comparison.png'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({'analysis_complete':True,'executed_proofs':sum(s['executed_proof'] is not None for s in steps),
        'raw_results_unchanged':True},indent=2))


if __name__=='__main__':main()
