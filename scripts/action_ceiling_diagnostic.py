"""Separate action discretization, measurement precision and oracle pair ambiguity."""
from datetime import datetime,timezone
from dataclasses import replace
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import ndtr

from history_refinement import digest,write_json
from finite_damage_guard import SyntheticSpecimen,forecast,fine_grid_audit,evaluate
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture,DriveState,cycle
from wmfracture.compliant_inference import PrefixModel,grid,equilibrium_cycle,damage
from wmfracture.damage_guard import FiniteGuard,feasible_bias_interval,action_risks
from wmfracture.action_ceiling import finite_action_limit,proxy_optimal_action,noiseless_history_manifold


def collect(root,config,law,fixture,cases):
    public,private,decisions = [],[],[]
    for index,case in enumerate(cases):
        for si,sigma in enumerate(config['gap_noise_sigma_m']):
            for bi,bias in enumerate(config['actual_bias_m']):
                context = f'ceiling-{index:03d}-{si}-{bi}'
                # Same standardized noise draws across sigmas AND policies.
                seed = config['seed']+index*1000+bi*100
                for policy in config['policies']:
                    specimen = SyntheticSpecimen(case['parameters'],bias,sigma,seed,law,fixture)
                    controller = FiniteGuard(law,fixture,config,sigma)
                    actions = [0.,config['weak_drive_m']]
                    observed = [specimen.observe(a) for a in actions]
                    identifier = context+'-'+policy
                    for turn in range(config['decision_rounds']):
                        prefix = {'actions_m':actions.copy(),'observed_gaps_m':observed.copy()}
                        action,record = controller.plan(actions,observed,policy,turn)
                        decision = {'episode_id':identifier,'context_id':context,'sigma_m':sigma,'public_prefix':prefix,**record}
                        decision['decision_sha256'] = hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()
                        decisions.append(decision)
                        actions.append(action);observed.append(specimen.observe(action))
                    public.append({'id':identifier,'context_id':context,'policy':policy,'sigma_m':sigma,
                        'actions_m':actions,'observed_gaps_m':observed})
                    private.append({'episode_id':identifier,'context_id':context,'case_id':case['id'],
                        'in_prior':case['in_prior'],'parameters':case['parameters'],'actual_bias_m':bias,
                        'trace':specimen.trace,'prepared_maximum_m':specimen.prepared_maximum})
        if index%5==0:
            print(f'CEILING_CASE {index+1}/{len(cases)}',flush=True)
    write_json(root/'public-episodes.json',public);write_json(root/'private-traces.json',private)
    write_json(root/'decisions.json',decisions)


def pair_diagnostic(root,settings,config,base,cases,law,fixture):
    """Privileged two-state references, NOT executable observation policies."""
    grouped = {}
    for case in cases:
        if case['family']!='matched_stiffness':
            continue
        energy,prep = case['parameters']
        _,maximum = equilibrium_cycle(0.,energy,prep,law,fixture)
        gap,maximum = equilibrium_cycle(maximum,energy,config['weak_drive_m'],law,fixture)
        ratio = round(float(1-damage(maximum,energy,law)),8)
        grouped.setdefault(ratio,[]).append((case,float(maximum),float(gap)))
    model = PrefixModel(law,fixture,config['parameter_bounds']);unit = grid(settings['audit_side'])
    prefix,pool_max = model.predict(unit,[0.,config['weak_drive_m']]);pool_energy = model.parameters(unit)[:,0]
    rows = []
    for ratio,group in grouped.items():
        for a,b in combinations(group,2):
            assert abs(a[2]-b[2]) < 1e-12
            energy = np.array([a[0]['parameters'][0],b[0]['parameters'][0]])
            maximum = np.array([a[1],b[1]])
            pair_prefix = np.array([[0.,a[2]],[0.,b[2]]])
            pair_limit,valid = finite_action_limit(maximum,maximum,energy,config['damage_cap'],config['actions_m'][-1],law,fixture)
            assert valid
            wide_limit,wide_valid = finite_action_limit(maximum,maximum,energy,config['damage_cap'],settings['oracle_wide_drive_m'],law,fixture)
            assert wide_valid
            manifold = noiseless_history_manifold(ratio,config['parameter_bounds'][0],config['parameter_bounds'][1],
                config['weak_drive_m'],config['damage_cap'],config['actions_m'][-1],law,fixture)
            assert manifold['valid']
            future_events = []
            for command in base['future_drives_m']:
                _,after = equilibrium_cycle(maximum,energy,command,law,fixture)
                future_events.append(bool((damage(after,energy,law)>=1-1e-12)[0] != (damage(after,energy,law)>=1-1e-12)[1]))
            for sigma in settings['pair_noise_sigma_m']:
                # Mean observations, zero bias, no realized noise: vary uncertainty
                # radii without conflating a particular noise draw or estimator.
                y = [0.,a[2]]
                oracle = proxy_optimal_action(pair_prefix,y,sigma,0.,[.5,.5],maximum,energy,pair_limit,law,fixture)
                pair_gap,_ = equilibrium_cycle(maximum,energy,oracle['action_m'],law,fixture)
                pair_signal = float(abs(np.diff(pair_gap)[0]))
                wide = proxy_optimal_action(pair_prefix,y,sigma,0.,[.5,.5],maximum,energy,wide_limit,law,fixture)
                wide_gap,_ = equilibrium_cycle(maximum,energy,wide['action_m'],law,fixture)
                wide_signal = float(abs(np.diff(wide_gap)[0]))
                ideal = proxy_optimal_action(pair_prefix,y,sigma,0.,[.5,.5],maximum,energy,manifold['one_step_limit_m'],law,fixture)
                ideal_gap,_ = equilibrium_cycle(maximum,energy,ideal['action_m'],law,fixture)
                ideal_signal = float(abs(np.diff(ideal_gap)[0]))
                for bias_bound in settings['pair_bias_bounds_m']:
                    mask,_,_ = feasible_bias_interval(prefix,y,sigma,config['noise_multiple'],bias_bound)
                    pool_limit,pool_valid = finite_action_limit(pool_max[mask],pool_max[mask],pool_energy[mask],
                        config['damage_cap'],config['actions_m'][-1],law,fixture)
                    restricted = proxy_optimal_action(pair_prefix,y,sigma,0.,[.5,.5],maximum,energy,pool_limit,law,fixture)
                    gap,_ = equilibrium_cycle(maximum,energy,restricted['action_m'],law,fixture)
                    signal = float(abs(np.diff(gap)[0]))
                    risk,_ = action_risks(maximum,maximum,energy,[restricted['action_m']],law,fixture)
                    rows.append({'pair':[a[0]['id'],b[0]['id']],'effective_stiffness_ratio':ratio,'sigma_m':sigma,
                        'bias_bound_m':bias_bound,'future_event_disagreements':sum(future_events),
                        'oracle_pair_limit_m':pair_limit,'oracle_pair_action_m':oracle['action_m'],
                        'oracle_pair_signal_mm':pair_signal*1000,
                        'oracle_pair_one_observation_bayes_error':float(ndtr(-pair_signal/(2*sigma))),
                        'wide_oracle_pair_limit_m':wide_limit,'wide_oracle_pair_signal_mm':wide_signal*1000,
                        'wide_oracle_pair_one_observation_bayes_error':float(ndtr(-wide_signal/(2*sigma))),
                        'exact_noiseless_manifold':manifold,'signal_under_exact_noiseless_manifold_mm':ideal_signal*1000,
                        'finite_pool_candidates':int(mask.sum()),'finite_pool_limit_m':pool_limit,'finite_pool_valid':pool_valid,
                        'pair_signal_under_pool_limit_mm':signal*1000,
                        'pair_one_observation_error_under_pool_limit':float(ndtr(-signal/(2*sigma))),
                        'true_pair_extra_damage_under_pool_action':float(risk.max()),
                        'true_pair_cap_violation_under_pool_action':bool(risk.max()>config['damage_cap']+1e-12)})
    write_json(root/'oracle-pair-diagnostic.json',{'rows':rows,'scope':
        'Privileged known two-state alternatives, known zero sensor bias, equal prior, one scalar Gaussian observation. Phi(-signal/(2*sigma)) is only this binary classification error; not an information bound for the full task, repeated adaptive experiments or physical reality. Full-pool sets use mean initial observations and finite parameter grids.'})
    return rows


def audit_counterexamples(root,settings,config,law,fixture,rows):
    """Privileged failure diagnosis after actions and forecasts are fixed."""
    failures = [r for r in rows if r['in_prior'] and r['true_cap_violation']]
    public = {e['id']:e for e in json.loads((root/'public-episodes.json').read_text())}
    private = {e['episode_id']:e for e in json.loads((root/'private-traces.json').read_text())}
    decisions = json.loads((root/'decisions.json').read_text());results = []
    replay = settings['counterexample_replay']
    for failure in failures:
        identifier = failure['episode_id'];e,p = public[identifier],private[identifier]
        energy,prep = p['parameters'];material = replace(law,fracture_energy=energy)
        expected = failure['true_extra_damage'];dynamic = []
        for dt in replay['dt_levels']:
            state = DriveState()
            cycle(state,material,fixture,prep,dt,**replay['cycle'])
            reference = None
            for index,command in enumerate(e['actions_m']):
                cycle(state,material,fixture,command,dt,**replay['cycle'])
                if index==1:
                    reference = material.damage(state.maximum_opening)
            extra = material.damage(state.maximum_opening)-reference
            assert abs(extra-expected)<replay['maximum_damage_difference']
            dynamic.append({'dt':dt,'extra_damage':extra,'difference_from_quasistatic':extra-expected,
                'maximum_opening_m':state.maximum_opening,'continuous_physical_time_s':state.time})
        probes = []
        for decision in [d for d in decisions if d['episode_id']==identifier]:
            prefix = decision['public_prefix'];baseline = p['trace'][1]['maximum_opening_m']
            before = p['trace'][1+decision['round']]['maximum_opening_m']
            actual_risk,_ = action_risks(np.array([before]),np.array([baseline]),np.array([energy]),[decision['action_m']],law,fixture)
            levels = []
            for side in replay['grid_sides']:
                model = PrefixModel(law,fixture,config['parameter_bounds']);unit = grid(side)
                prediction,maximum = model.predict(unit,prefix['actions_m'])
                _,reference = model.predict(unit,[0.,config['weak_drive_m']])
                mask,_,_ = feasible_bias_interval(prediction,prefix['observed_gaps_m'],e['sigma_m'],config['noise_multiple'],config['bias_bound_m'])
                risks,_ = action_risks(maximum[mask],reference[mask],model.parameters(unit)[mask,0],[decision['action_m']],law,fixture)
                levels.append({'side':side,'feasible_samples':int(mask.sum()),
                    'sample_maximum_damage':float(risks.max()) if mask.any() else None})
            bounds = np.asarray(config['parameter_bounds'])
            true_unit = ((np.asarray(p['parameters'])-bounds[:,0])/np.ptp(bounds,axis=1))[None,:]
            exact,_ = PrefixModel(law,fixture,bounds).predict(true_unit,prefix['actions_m'])
            compatible,_,_ = feasible_bias_interval(exact,prefix['observed_gaps_m'],e['sigma_m'],config['noise_multiple'],config['bias_bound_m'])
            probes.append({'round':decision['round'],'action_m':decision['action_m'],
                'true_parameters_satisfy_current_constraints':bool(compatible[0]),
                'true_cumulative_risk_of_chosen_action':float(actual_risk[0,0]),'grid_refinements':levels})
        results.append({'episode_id':identifier,'parameters':p['parameters'],'actual_bias_m':p['actual_bias_m'],
            'sigma_m':e['sigma_m'],'quasistatic_extra_damage':expected,
            'noise_and_bias_within_assumed_bounds':failure['actual_noise_and_bias_within_assumed_bounds'],
            'dynamic_replay':dynamic,'decision_diagnostics':probes})
    result = {'counterexamples':results,'scope':'Post-hoc diagnosis only. Replays same remote preparation/actions without state resets; no privileged information fed back to policy. Grid maxima remain empirical samples.'}
    write_json(root/'counterexample-audit.json',result)
    return result


def summarize(root,report,pairs):
    lines = ['# 행동 간격·관측 정밀도·두 상태 진단','',
        '기존 26개 합성 상태에서 새 paired noise를 생성했다. 각 sigma 조건에서 실제 생성 잡음과 추정기의 sigma를 같이 바꿨다. 잡음은 유지한 채 허용 오차만 줄이는 실험이 아니다. 같은 표준정규 난수를 정책과 정밀도 조건에 공유하므로 독립 반복으로 세지 않는다.','',
        'coarse는 기존 7개 입력, continuous는 같은 유한 후보에서 누적 손상 한도까지의 입력 범위를 계산하고 관측 차이 proxy를 전역 최대화한다. 위험 후보는 두 정책 모두 161²다. 파라미터 영역은 여전히 유한 격자다.','',
        '## 범위 안 24개 상태, 실제 편향 +0.030mm, 2단계 폐루프','',
        '| 실제 잡음 sigma (mm) | 정책 | 미래 RMSE (mm) | 실제 손상 한도 위반 | 최대 추가 D | 321² 금지 선택 수 / 48 | 감사 후보 최대 추가 D | 빈 집합 단계 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for sigma in report['settings']['gap_noise_sigma_m']:
        for policy in report['settings']['policies']:
            r = next(r for r in report['groups'] if r['in_prior'] and r['sigma_m']==sigma and r['actual_bias_m']==.00003 and r['policy']==policy)
            audit = str(r['fine_grid_forbidden_steps']) if policy!='rest_only' else '대상 아님'
            fine_damage = '대상 아님' if r['fine_grid_maximum_sample_damage'] is None else f"{r['fine_grid_maximum_sample_damage']:.5f}"
            lines.append(f"| {sigma*1000:.3f} | {policy} | {r['gap_rmse_mm']:.4f} | {r['cap_violation_fraction']:.1%} | {r['maximum_true_extra_damage']:.5f} | {audit} | {fine_damage} | {r['empty_set_steps']} |")
    lines += ['', '## 실제 한도 위반도 발생했다','',
        f"범위 안 전체 조건에서 연속 입력 정책의 실제 한도 위반은 {report['in_prior_true_cap_violations']}회였다. 대표 +0.030mm 조건 표만으로 안전성을 판단하면 안 된다. counterexample-audit.json에 참 파라미터의 관측 제약 부합, 후보 격자별 최댓값, 기존 동적 계산기의 같은 준비/행동 재생을 남겼다. 성공한 예측 오차를 같은 연속 안전 조건에서의 개선이라고 주장하지 않는다.",'']
    lines += ['', '## 정확히 최대화한 것은 관측 차이 proxy다','',
        '후보별 다음 출력은 입력에 대해 구간별 affine이다. 조건부 센서 offset을 더해도 affine 구조는 유지된다. 후보 분산은 각 구간에서 convex quadratic이므로 최대는 끝점이나 모드 경계에서 생긴다. 정렬한 경계에서 가중 moment를 갱신하는 표준 계산으로 후보 수의 제곱 크기 rollout을 피했다.','',
        '손상 한도 입력은 기존 bilinear D 법칙의 역함수로 계산한다. 분리는 작은 수치 여유로 배제한다. 이는 현재 안정 단조 스칼라 모형과 유한 후보에 한정된다. 진짜 mutual information·미래 오차 감소·연속 물리 상태의 안전 상한은 아니다.','',
        '## 정답 두 상태만 안다는 특권 진단','',
        '같은 약한 반응을 보이는 18개 설계 상태 쌍을 사용했다. 초기 관측은 잡음이 실현된 값이 아닌 평균값으로 고정했다. 두 상태만 가능한 경우와 전체 사전 격자를 유지하는 경우의 허용 입력·반응 차이를 비교한다. 이 정보는 실제 정책에 주지 않았다. 아래 주 표는 기존 입력 범위 1.85mm 이하다. 각 통계 열을 따로 요약했으므로 신호 중앙값을 오류 식에 넣으면 오류 중앙값과 다를 수 있다.','',
        '| sigma (mm) | 전체 집합의 bias 범위 (mm) | 두 상태만 아는 경우: 반응 차이 중앙값 (mm) | 전체 집합 제한: 차이 중앙값 (mm) | 두 상태 단일 관측 오류 중앙값 | 전체 제한 오류 중앙값 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for sigma in report['diagnostic_settings']['pair_noise_sigma_m']:
        for bound in report['diagnostic_settings']['pair_bias_bounds_m']:
            selected = [r for r in pairs if r['sigma_m']==sigma and r['bias_bound_m']==bound]
            med = lambda key:float(np.median([r[key] for r in selected]))
            lines.append(f"| {sigma*1000:.3f} | ±{bound*1000:.3f} | {med('oracle_pair_signal_mm'):.5f} | {med('pair_signal_under_pool_limit_mm'):.5f} | {med('oracle_pair_one_observation_bayes_error'):.3f} | {med('pair_one_observation_error_under_pool_limit'):.3f} |")
    lines += ['', '입력 최대치를 2.30mm로 넓힌 특권 두 상태 진단도 별도로 계산했다. 이는 실행 정책의 행동 범위를 사후에 바꾼 결과가 아니다.','',
        '| 입력 범위 | 두 상태 최적 반응 차이 중앙값 (mm) | 동일 반응 쌍 / 18 |', '| --- | ---: | ---: |']
    selected = [r for r in pairs if r['sigma_m']==.00002 and r['bias_bound_m']==.00015]
    for label,key in [('≤1.85mm','oracle_pair_signal_mm'),('≤2.30mm','wide_oracle_pair_signal_mm')]:
        lines.append(f"| {label} | {np.median([r[key] for r in selected]):.5f} | {sum(r[key]<1e-9 for r in selected)} |")
    lines += ['', '입력 범위가 제한돼서 생기는 동일 반응을 손상 한도의 근본 한계라고 부르지 않는다. 현재 폐루프 정책은 1.85mm 범위 그대로이며 넓은 범위는 진단용이다.','',
        '## 초기에 약한 반응을 완벽하게 알아도 남는 모호성','',
        '알려진 bias와 정확한 초기 unloading 강성 q를 가정하면 D=1-q를 알 수 있다. 하지만 Gc와 과거 최대 벌어짐은 하나로 정해지지 않고 연속 곡선을 이룬다. 이 제한된 무잡음 곡선에서는 손상 한도 입력이 Gc에 따라 증가하므로 최소 Gc 끝점에서 한 번의 행동 허용치를 해석적으로 계산할 수 있다. 일반적인 잡음 있는 2D 영역의 보장은 아니다.','',
        '| 강성 비율 q | 정확한 초기 반응으로 허용되는 입력 상한 (mm) | 그 범위에서 다음 반응이 같은 설계 쌍 / 6 |',
        '| --- | ---: | ---: |']
    for ratio in sorted({r['effective_stiffness_ratio'] for r in pairs}):
        selected = [r for r in pairs if r['effective_stiffness_ratio']==ratio and r['sigma_m']==.00002 and r['bias_bound_m']==.00015]
        limit = selected[0]['exact_noiseless_manifold']['one_step_limit_m']*1000
        indistinguishable = sum(r['signal_under_exact_noiseless_manifold_mm'] < 1e-9 for r in selected)
        lines.append(f'| {ratio:.2f} | {limit:.5f} | {indistinguishable} |')
    lines += ['', '이는 다음 한 번의 반응에 대한 진단이다. 다른 저에너지 가설을 먼저 배제한 뒤 더 큰 입력을 허용하는 다단계 전략까지 불가능하다는 뜻은 아니다. 단순한 행동 세분화·큰 encoder만으로 모든 모호성이 사라진다고 볼 수 없다는 근거로 한정한다.','',
        '## 해석의 범위','',
        '단일 관측 오류 Phi(-|차이|/(2 sigma))는 두 상태·동일 사전확률·알려진 센서 bias·Gaussian scalar 관측에서의 이진 판별 오류다. 전체 연구의 정보 한계나 여러 번의 적응 관측에 대한 하한이 아니다. 18쌍도 독립 시편이 아니며 재료/이력 설계 상태들을 공유한다. 미래 분리 결과가 다른 쌍과 아닌 쌍도 JSON에서 구분한다.','',
        '더 낮은 sigma는 가상 조건이며 실제 카메라에서 달성했다고 하지 않는다. 관측 집합이 비거나 격자 간 판정이 바뀌면 정밀도 향상의 효과와 후보 해상도 부족을 구분해야 한다. 기존 결과를 덮어쓰지 않았고 새 하이퍼파라미터를 시험 정답으로 선택하지 않았다.','']
    (root/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(15,4.3),constrained_layout=True)
    for policy in report['settings']['policies']:
        selected = [next(r for r in report['groups'] if r['in_prior'] and r['sigma_m']==sigma and r['actual_bias_m']==.00003 and r['policy']==policy) for sigma in report['settings']['gap_noise_sigma_m']]
        x = np.array(report['settings']['gap_noise_sigma_m'])*1000
        axes[0].plot(x,[r['gap_rmse_mm'] for r in selected],'o-',label=policy)
        all_bias = [[r for r in report['groups'] if r['in_prior'] and r['sigma_m']==sigma and r['policy']==policy] for sigma in report['settings']['gap_noise_sigma_m']]
        rates = [100*sum(r['cap_violation_fraction']*r['episodes'] for r in group)/sum(r['episodes'] for r in group) for group in all_bias]
        axes[1].plot(x,rates,'o-',label=policy)
    sigma = .00002
    p = [r for r in pairs if r['sigma_m']==sigma and r['bias_bound_m']==.00015]
    axes[2].scatter([r['oracle_pair_signal_mm'] for r in p],[r['pair_signal_under_pool_limit_mm'] for r in p])
    maximum = max(r['oracle_pair_signal_mm'] for r in p);axes[2].plot([0,maximum],[0,maximum],'k--',alpha=.4)
    axes[2].set(xlabel='Known-pair best signal (mm)',ylabel='Signal under full finite-pool limit (mm)',title='Privileged diagnostic; sigma=0.020 mm')
    axes[0].set(xlabel='Actual measurement sigma (mm)',ylabel='Future hold-gap RMSE (mm)',xscale='log',title='True bias +0.030 mm; own policy truth')
    axes[1].set(xlabel='Actual measurement sigma (mm)',ylabel='True cap violation fraction (%)',xscale='log',ylim=(0,3),title='Both biases: 48 in-prior episodes per sigma')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)


def main():
    project = Path(os.environ['WM_PROJECT_ROOT'])
    settings = json.loads((project/'config/action-ceiling-diagnostic.json').read_text())
    config = json.loads((project/settings['guard_config']).read_text())
    config.update({k:settings[k] for k in ('grid_side','gap_noise_sigma_m','actual_bias_m','policies','seed')})
    source = Path(json.loads((project/settings['source_report']).read_text())['run_directory'])
    base = json.loads((source/'config.json').read_text());cases = json.loads((source/'private-cases.json').read_text())
    original = {n:digest(source/n) for n in ('observations.json','private-cases.json','report.json','forecasts.json')}
    root = Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-action-ceiling-diagnostic')
    root.mkdir(parents=True);print(f'OUTPUT={root}',flush=True)
    write_json(root/'config.json',config);write_json(root/'diagnostic-settings.json',settings);write_json(root/'physics-config.json',base)
    sources = [Path(__file__),project/'wmfracture/action_ceiling.py',project/'wmfracture/damage_guard.py',
        project/'wmfracture/compliant_inference.py',project/'wmfracture/measurement_bias.py',
        project/'wmfracture/cohesive.py',project/'wmfracture/compliant_fixture.py',project/'scripts/finite_damage_guard.py']
    for p in sources:
        target = root/'source'/p.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in sources})
    law = InterfaceLaw(**json.loads((project/base['law_config']).read_text())['law']);fixture = Fixture(**base['fixture'])
    shutil.copy2(project/base['law_config'],root/'law-config.json')
    started = time.perf_counter();collect(root,config,law,fixture,cases)
    seals = forecast(root,config,base,law,fixture)
    audit = fine_grid_audit(root,config,law,fixture,side=settings['audit_side'])
    groups,rows = evaluate(root,config,law,fixture,seals)
    for group in groups:
        episode_ids = {r['episode_id'] for r in rows if (r['in_prior'],r['sigma_m'],r['actual_bias_m'],r['policy'])
            == (group['in_prior'],group['sigma_m'],group['actual_bias_m'],group['policy'])}
        values = [r['fine_finite_sample_maximum_damage'] for r in audit['records']
            if r['episode_id'] in episode_ids and r['fine_finite_sample_maximum_damage'] is not None]
        group['fine_grid_maximum_sample_damage'] = max(values) if values else None
    pairs = pair_diagnostic(root,settings,config,base,cases,law,fixture)
    failures = audit_counterexamples(root,settings,config,law,fixture,rows)
    assert seals == {n:digest(root/n) for n in seals}
    assert original == {n:digest(source/n) for n in original}
    report = {'run_directory':str(root),'source_run':str(source),'settings':config,'diagnostic_settings':settings,
        'groups':groups,'physical_cases':len(cases),'policy_episodes':len(rows),'pair_comparisons':len(pairs),
        'total_seconds':time.perf_counter()-started,'source_files_unchanged':True,'source_sha256':original,'seals':seals,
        'scope':settings['scope'],'continuous_parameter_safety_bound_validated':False,
        'restricted_noiseless_one_step_manifold_analyzed':True,'new_method_novelty_validated':False}
    report['in_prior_true_cap_violations'] = sum(r['in_prior'] and r['true_cap_violation'] for r in rows)
    report['dynamic_counterexamples_replayed'] = len(failures['counterexamples'])
    write_json(root/'report.json',report);summarize(root,report,pairs)
    out = project/'artifacts/action-ceiling-diagnostic';out.mkdir(parents=True,exist_ok=True)
    for name in ('report.json','summary.md','comparison.png','config.json','diagnostic-settings.json','code-sha256.json',
        'evaluation-rows.json','decisions.json','fine-grid-audit.json','oracle-pair-diagnostic.json','sealed-predictions.json','forecast-cost.json','counterexample-audit.json'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('settings','diagnostic_settings','groups','source_sha256','seals')},indent=2),flush=True)


if __name__=='__main__':
    main()
