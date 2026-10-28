"""Streaming grid-refinement audit and readable report of an existing run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest, write_json
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import PrefixModel, grid, equilibrium_cycle, damage, log_likelihood, normalize


def refinement(root, config, levels=(81, 161, 321, 641)):
    project = Path(os.environ['WM_PROJECT_ROOT'])
    law = InterfaceLaw(**json.loads((project/config['law_config']).read_text())['law'])
    fixture = Fixture(**config['fixture'])
    episodes = json.loads((root/'observations.json').read_text())
    previous, result = None, []
    hashes = {name:digest(root/name) for name in ('posteriors.npz','fit-packets.json','forecasts.json','private-cases.json')}
    for side in levels:
        tick = time.perf_counter()
        model = PrefixModel(law, fixture, config['parameter_bounds'])
        unit = grid(side); energy = model.parameters(unit)[:, 0]
        lookup = {}
        for protocol, actions in config['protocols'].items():
            prediction, maximum = model.predict(unit, actions)
            futures = []
            for command in config['future_drives_m']:
                gap, after = equilibrium_cycle(maximum, energy, command, law, fixture)
                futures.append((gap, damage(after, energy, law) >= 1-1e-12))
            for episode in episodes:
                if episode['protocol'] != protocol:
                    continue
                weights = normalize(log_likelihood(prediction, episode['observed_gaps_m'], episode['sigma_m']))
                lookup[episode['id']] = [[float(np.sum(weights*gap)), float(np.sum(weights*event))] for gap,event in futures]
        current = {'side':side, 'candidates':side*side, 'seconds':time.perf_counter()-tick,
            'cost':dict(model.cost), 'predictions':lookup}
        if previous is not None:
            differences = []
            for episode in episodes:
                a, b = np.asarray(previous['predictions'][episode['id']]), np.asarray(lookup[episode['id']])
                differences.append({'episode_id':episode['id'], 'scenario':episode['scenario'],
                    'sigma_m':episode['sigma_m'], 'protocol':episode['protocol'],
                    'max_mean_gap_difference_mm':float(np.max(np.abs(a[:, 0]-b[:, 0]))*1000),
                    'max_separation_mass_difference':float(np.max(np.abs(a[:, 1]-b[:, 1])))})
            current['changes_from_previous'] = differences
            current['maximum_mean_gap_difference_mm'] = max(r['max_mean_gap_difference_mm'] for r in differences)
            current['maximum_separation_mass_difference'] = max(r['max_separation_mass_difference'] for r in differences)
        result.append(current); previous = current
        print(f'GRID_REFINEMENT side={side} seconds={current["seconds"]:.2f}', flush=True)
    assert hashes == {name:digest(root/name) for name in hashes}
    report = {'levels':list(levels), 'stages':[3], 'records':result,
        'original_predictions_unchanged':True,
        'scope':'Final observation stage only, same episodes. Successive grids are an empirical quadrature check, not a proof of posterior convergence or state containment.'}
    write_json(root/'additional-grid-refinement.json', report)
    return report


def readable(root, config, refinement_report):
    report = json.loads((root/'report.json').read_text())
    groups = report['groups']
    lines = ['# 외부 데이터 없는 합성 추정 실험', '',
        '현재 모형의 천천히 가하는 하중에서 벌어짐 관측을 합성했다. Houdini·Unreal·촬영·GPU·신경망을 사용하지 않는다.', '',
        '26개 물리 사례(사전 범위 안 24개, 밖 2개)를 2개 입력 프로토콜·3개 잡음·2개 관측 모델 조건으로 비교했다. 312개 관측 조건은 312개 독립 시편이 아니다. SMC 세 seed도 독립 물리 사례를 늘리지 않는다.', '',
        '## 비교 방법', '',
        '- grid7 hard: 49개 고정 후보에서 매 관측의 ±3σ를 만족하는 후보만 유지. 빈 집합은 그대로 기록하며 대체 후보를 몰래 넣지 않는다.',
        '- grid7/41/81 soft: 각각 49/1,681/6,561개 후보에 같은 Gaussian likelihood를 적용한다.',
        '- SMC128 resample: 128개 연속 파라미터 후보, ESS 기준 재표집.',
        '- SMC128 move: 같은 초기 후보와 재표집에 관측 단계마다 2회 Metropolis 이동을 추가. 제안마다 준비부터 현재까지 전체 이력을 재생한다.', '',
        '사전분포는 Gc와 관측하지 못한 과거 구동부 최대 변위에 균등하다. 숨은 손상도에 균등한 사전분포가 아니다. Hard filter 후보 개수 비율을 분리 확률로 평가하지 않는다.', '',
        '## 대표 조건: 범위 안 24개 사례, 관측 σ=0.020mm, 3회 관측 후', '',
        '세 미래 입력의 유지 시점 벌어짐을 평가했다. 경험적 95% 포함률은 각 확률 방법의 명목 구간 안에 정답이 든 비율이며 보장이 아니다. Hard filter에는 95% 확률 구간을 붙이지 않고, 전체 유지 집합의 포함률을 별도로 기록했다. Hard filter의 점 예측은 후보들의 단순 산술 평균이며 비어 있지 않은 조건에서만 오차를 계산하므로 빈 집합 비율도 함께 읽어야 한다.', '',
        '| 관측 조건 | 입력 | 방법 | 미래 벌어짐 RMSE (mm) | 명목 95% 포함률 | 분리 Brier | 빈 집합 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    names = {'matched':'오차 모델 일치', 'unmodelled_constant_bias':'미반영 +0.030mm bias',
        'repeated_weak':'약한 입력 반복', 'fixed_probe':'고정 탐색 입력'}
    for scenario in config['observation_scenarios']:
        for protocol in config['protocols']:
            for method in ('grid7_hard','grid7_soft','smc128_resample','smc128_move','grid81_soft'):
                r = next(r for r in groups if r['in_prior'] and r['scenario']==scenario and r['protocol']==protocol
                    and r['sigma_m']==.00002 and r['stage']==3 and r['method']==method)
                brier = '—' if r['separation_brier'] is None else f"{r['separation_brier']:.4f}"
                coverage = '—' if r['empirical_95_coverage'] is None else f"{r['empirical_95_coverage']:.1%}"
                lines.append(f"| {names[scenario]} | {names[protocol]} | {method} | {r['gap_rmse_mm']:.4f} | {coverage} | {brier} | {r['empty_fraction']:.1%} |")
    lines += ['', '## 계산 비용', '',
        '최종 단계까지 SMC resample은 384회, resample-move는 1,152회의 파라미터-prefix 계산을 수행했다. 49개 고정 후보보다 계산량이 크다. 동일 예산 우위나 새 알고리즘의 개선으로 해석하지 않는다. 각 prefix 계산에는 준비와 그때까지의 모든 행동 재생이 포함되므로 root 계산 수도 JSON에 별도로 기록한다.',
        '격자 bank는 여러 사례와 잡음에 재사용했다. library-cost.json은 실제 공유 bank 비용, fit-packets.json은 각 방법을 독립 실행한다고 가정한 논리 비용이다. 전체 fit 시간만으로 SMC 대 격자의 속도 우위를 주장하지 않는다.', '',
        '## 해석식과 기존 동적 계산기 비교', '',
        f"선택한 4개 사례·2개 시간 간격에서 유지 시점 벌어짐의 최대 차이는 {report['dynamic_audit']['maximum_hold_gap_error_m']*1000:.9f}mm였다. 실제 이력 최대값의 차이는 {report['dynamic_audit']['maximum_post_cycle_history_error_m']*1000:.9f}mm였다. 선택한 느린 안정 분기의 수치 검증이며 빠른 충격이나 실물 검증은 아니다.", '',
        '## 손상 비용', '',
        f"고정 탐색의 수동 대조군 대비 추가 D 최댓값은 {report['maximum_true_probe_extra_damage']:.6f}였다. 앞서 고른 두 예제에서 약 0.0125였다는 결과가 전체 상태의 작은 손상으로 일반화되지 않았다. 이 정책을 안전하다고 부르지 않는다.", '',
        '## 조밀한 격자도 정답 보장은 아니다', '',
        f"처음 41×41과 81×81 비교에서 평균 미래 벌어짐의 최대 차이는 {report['grid_refinement_max_mean_gap_mm']:.4f}mm, 분리 질량 차이는 {report['grid_refinement_max_separation_mass_difference']:.4f}였다. 아래 추가 검사는 관측 3단계에서만 수행했고, 위 원래 표를 수정하지 않았다.", '',
        '| 격자 비교 | 최대 예측 평균 차이 (mm) | 최대 분리 질량 차이 |', '| --- | ---: | ---: |']
    for r in refinement_report['records'][1:]:
        previous = refinement_report['levels'][refinement_report['levels'].index(r['side'])-1]
        lines.append(f"| {previous}² → {r['side']}² | {r['maximum_mean_gap_difference_mm']:.5f} | {r['maximum_separation_mass_difference']:.5f} |")
    lines += ['', '수렴 차이는 조건별로 additional-grid-refinement.json에 남겼다. 작은 관측 잡음, 미반영 bias, 사전 범위 밖 상태를 분리해서 읽어야 한다. 두 격자 사이 차이가 작아도 연속 영역의 인증된 경계는 아니다.', '',
        '## 이번 결과로 할 수 있는 판단', '',
        '1. 고정 후보를 제거하기만 하는 방식 대신 표준 resample-move를 강한 기준선으로 사용한다. 현재 결과로 새 압축법의 기여를 주장하지 않는다.',
        '2. 반복 관측은 잡음을 줄일 수 있지만 관측 자체가 구별하지 못하는 물성/이력을 유일하게 만들지는 않는다.',
        '3. 강한 탐색이 예측을 개선하더라도 비가역 손상까지 계산해야 한다. 다음 단계는 전체 후보 위험 검사와 정보 목적함수를 분리하는 것이다.',
        '4. 관측 bias를 모델링하지 않으면 조밀한 후보도 명목 포함률을 잃는다. 다음 단계에는 공통 bias를 추정하는 단순 기준선이 필요하다.',
        '5. 2개 범위 밖 사례는 스트레스 사례일 뿐 모집단 성능을 추정하는 표본이 아니다. 참 상태가 사전 범위 밖이면 후보 이동만으로 해결할 수 없다.', '',
        '안정적인 단일 접합면에서는 단조 완전 분리의 구동 변위 기준이 재료의 최종 벌어짐으로 정해진다. 이번 미래 분리 평가는 재료/이력의 결합 모호성을 다루며, 같은 재료에서 이력만 바꿔 완전 분리 기준이 달라짐을 보인 실험이 아니다.', '',
        '이 결과는 새 방법·일반 3D 파괴·학습 월드모델의 검증이 아니다. 방법 개발 전에 병목을 분리하는 개발 진단이다.', '']
    (root/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    methods = ('grid7_soft','smc128_resample','smc128_move','grid81_soft')
    for method in methods:
        selected = [next(r for r in groups if r['in_prior'] and r['scenario']=='matched' and r['sigma_m']==.00002
            and r['protocol']=='fixed_probe' and r['stage']==stage and r['method']==method) for stage in (1, 2, 3)]
        axes[0].plot([1,2,3], [r['gap_rmse_mm'] for r in selected], 'o-', label=method)
        axes[1].plot([1,2,3], [r['empirical_95_coverage'] for r in selected], 'o-', label=method)
    physical = json.loads((root/'intervention-costs.json').read_text())
    # One representative episode per physical case, not all repeated noise settings.
    episodes = json.loads((root/'observations.json').read_text())
    representative = {}
    for episode in episodes:
        if episode['protocol']=='fixed_probe':
            representative.setdefault(episode['case_id'],episode['id'])
    costs = [next(r['extra_damage_over_repeated_weak'] for r in physical if r['episode_id']==episode and r['stage']==3)
        for episode in representative.values()]
    axes[2].bar(np.arange(len(costs)), costs)
    axes[2].axhline(.02, color='r', linestyle='--', label='Earlier illustrative cap 0.02')
    axes[0].set(xlabel='Observation stage',ylabel='Future hold-gap RMSE (mm)',title='Matched model; sigma=0.020 mm')
    axes[1].set(xlabel='Observation stage',ylabel='Empirical nominal-95% coverage',ylim=(0,1.03),title='No calibration guarantee')
    axes[2].set(xlabel='Physical case index (26 cases)',ylabel='Extra damage over repeated weak',title='Fixed probes can add substantial damage')
    for axis in axes:
        axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.savefig(root/'comparison.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run/'config.json').read_text())
    result = refinement(args.run, config)
    readable(args.run, config, result)
    out = Path(os.environ['WM_PROJECT_ROOT'])/'artifacts/synthetic-filter-benchmark'
    for name in ('summary.md','comparison.png','additional-grid-refinement.json'):
        shutil.copy2(args.run/name, out/name)


if __name__ == '__main__':
    main()
