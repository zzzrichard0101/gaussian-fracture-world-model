"""Synthetic closed loop: finite full-pool guard vs subset and witness baselines."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest, write_json
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import PrefixModel, grid, equilibrium_cycle, damage, normalize, weighted_quantile
from wmfracture.measurement_bias import GaussianObservation
from wmfracture.damage_guard import FiniteGuard,feasible_bias_interval,action_risks,admissibility


class SyntheticSpecimen:
    """Private data generator; controller receives only observe()'s scalar."""
    def __init__(self, parameters, bias, sigma, seed, law, fixture):
        self.energy,self.preparation = parameters
        self.bias,self.sigma,self.rng = bias,sigma,np.random.default_rng(seed)
        self.law,self.fixture = law,fixture
        _,self.maximum = equilibrium_cycle(0.,self.energy,self.preparation,law,fixture)
        self.prepared_maximum = float(self.maximum)
        self.trace = []

    def observe(self, action):
        gap,self.maximum = equilibrium_cycle(self.maximum,self.energy,action,self.law,self.fixture)
        error = float(self.rng.normal(0,self.sigma))
        y = float(gap)+self.bias+error
        self.trace.append({'action_m':action,'true_gap_m':float(gap),'maximum_opening_m':float(self.maximum),
            'damage':float(damage(self.maximum,self.energy,self.law)),'noise_m':error,'observed_gap_m':y})
        return y


def collect(root,settings,cases,law,fixture):
    public,private,decisions = [],[],[]
    for case_index,case in enumerate(cases):
        for sigma_index,sigma in enumerate(settings['gap_noise_sigma_m']):
            for bias_index,bias in enumerate(settings['actual_bias_m']):
                context_id = f'context-{case_index:03d}-{sigma_index}-{bias_index}'
                seed = settings['seed']+case_index*1000+sigma_index*10+bias_index
                for policy in settings['policies']:
                    episode_id = context_id+'-'+policy
                    specimen = SyntheticSpecimen(case['parameters'],bias,sigma,seed,law,fixture)
                    controller = FiniteGuard(law,fixture,settings,sigma)
                    actions = [0.,settings['weak_drive_m']]
                    y = [specimen.observe(action) for action in actions]
                    for turn in range(settings['decision_rounds']):
                        before = {'actions_m':actions.copy(),'observed_gaps_m':y.copy()}
                        action,record = controller.plan(actions,y,policy,turn)
                        # Hash decision before the generator provides the next result.
                        decision = {'episode_id':episode_id,'context_id':context_id,
                            'public_prefix':before,'sigma_m':sigma,**record}
                        decision['decision_sha256'] = hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()
                        decisions.append(decision)
                        actions.append(action);y.append(specimen.observe(action))
                    public.append({'id':episode_id,'context_id':context_id,'policy':policy,'sigma_m':sigma,
                        'actions_m':actions,'observed_gaps_m':y})
                    private.append({'episode_id':episode_id,'context_id':context_id,'case_id':case['id'],
                        'in_prior':case['in_prior'],'parameters':case['parameters'],'actual_bias_m':bias,
                        'prepared_maximum_m':specimen.prepared_maximum,'trace':specimen.trace})
        if case_index % 5 == 0:
            print(f'GUARD_CASE {case_index+1}/{len(cases)}',flush=True)
    # Same noise draws and same full checks imply identical full/witness trajectories.
    indexed = {(e['context_id'],e['policy']):e for e in public}
    for context_id in {e['context_id'] for e in public}:
        full,witness = indexed[(context_id,'full_guard')],indexed[(context_id,'witness_guard')]
        assert full['actions_m']==witness['actions_m'] and full['observed_gaps_m']==witness['observed_gaps_m']
    write_json(root/'public-episodes.json',public);write_json(root/'private-traces.json',private)
    write_json(root/'decisions.json',decisions)


def forecast(root,settings,base,law,fixture):
    # Read public actions and observations only; no private-traces file.
    public = json.loads((root/'public-episodes.json').read_text())
    model = PrefixModel(law,fixture,settings['parameter_bounds']);unit = grid(settings['grid_side'])
    energy = model.parameters(unit)[:,0]
    archive,forecasts = {'unit':unit},[]
    for index,e in enumerate(public):
        prediction,maximum = model.predict(unit,e['actions_m'])
        weights = normalize(GaussianObservation('shared',settings['bias_sigma_m'])(prediction,e['observed_gaps_m'],e['sigma_m']))
        archive[f'weights{index}'] = weights
        futures = []
        for command in base['future_drives_m']:
            gap,after = equilibrium_cycle(maximum,energy,command,law,fixture)
            futures.append({'drive_m':command,'mean_gap_m':float(weights@gap),
                'q025_gap_m':weighted_quantile(gap,weights,.025),'q975_gap_m':weighted_quantile(gap,weights,.975),
                'separation_mass':float(weights@(damage(after,energy,law) >= 1-1e-12))})
        forecasts.append({'episode_id':e['id'],'weight_key':f'weights{index}','futures':futures})
    np.savez_compressed(root/'posteriors.npz',**archive);write_json(root/'forecasts.json',forecasts)
    write_json(root/'forecast-cost.json',{**model.cost,
        'future_action_candidate_queries':len(public)*len(base['future_drives_m'])*len(unit),
        'scope':'Forecast stage cost, separate from controller planning and numerical audits.'})
    seals = {name:digest(root/name) for name in ('public-episodes.json','decisions.json','posteriors.npz','forecasts.json')}
    write_json(root/'sealed-predictions.json',seals)
    return seals


def fine_grid_audit(root,settings,law,fixture,side=161):
    # Post-selection numerical audit only; never fed back into the controller.
    unit = grid(side);model = PrefixModel(law,fixture,settings['parameter_bounds'])
    energy = model.parameters(unit)[:,0]
    _,baseline = model.predict(unit,[0.,settings['weak_drive_m']])
    cache,records = {},[]
    decisions = json.loads((root/'decisions.json').read_text())
    for decision in decisions:
        content = {k:v for k,v in decision.items() if k != 'decision_sha256'}
        assert decision['decision_sha256'] == hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
    guarded = {'top8_guard','full_guard','witness_guard','full_guard_info8','continuous_guard'}
    action_queries = 0
    for decision in decisions:
        if decision['policy'] not in guarded:
            continue
        prefix = decision['public_prefix'];actions = tuple(prefix['actions_m'])
        if actions not in cache:
            cache[actions] = model.predict(unit,actions)
        prediction,maximum = cache[actions]
        mask,_,_ = feasible_bias_interval(prediction,prefix['observed_gaps_m'],decision['sigma_m'],
            settings['noise_multiple'],settings['bias_bound_m'])
        risks,failures = action_risks(maximum[mask],baseline[mask],energy[mask],[decision['action_m']],law,fixture)
        action_queries += risks.size
        records.append({'episode_id':decision['episode_id'],'round':decision['round'],'policy':decision['policy'],
            'coarse_empty':decision['empty_set'],'fine_candidates':int(mask.sum()),
            'fine_finite_sample_maximum_damage':float(risks.max()) if mask.any() else None,
            'fine_selected_action_forbidden':bool(not admissibility(risks,failures,settings['damage_cap'])[0]) if mask.any() else None})
    result = {'coarse_side':settings['grid_side'],'fine_side':side,'records':records,
        'cost':{**model.cost,'action_candidate_queries':action_queries},
        'scope':'Post-selection empirical selected-action audit. A finite refinement cannot certify a continuous domain.'}
    write_json(root/'fine-grid-audit.json',result)
    return result


def evaluate(root,settings,law,fixture,seals):
    public = {e['id']:e for e in json.loads((root/'public-episodes.json').read_text())}
    private = {e['episode_id']:e for e in json.loads((root/'private-traces.json').read_text())}
    decisions = json.loads((root/'decisions.json').read_text())
    forecasts = json.loads((root/'forecasts.json').read_text())
    fine = json.loads((root/'fine-grid-audit.json').read_text())['records']
    rows = []
    for prediction in forecasts:
        e,p = public[prediction['episode_id']],private[prediction['episode_id']]
        trace = p['trace'];reference = trace[1]['damage'];energy = p['parameters'][0]
        actual_extra = trace[-1]['damage']-reference
        decision = [d for d in decisions if d['episode_id']==e['id']]
        finer = [d for d in fine if d['episode_id']==e['id']]
        true_gaps = np.array([[r['true_gap_m'] for r in trace]])
        compatible,_,_ = feasible_bias_interval(true_gaps,e['observed_gaps_m'],e['sigma_m'],
            settings['noise_multiple'],settings['bias_bound_m'])
        future_errors,coverages,briers,widths,scores = [],[],[],[],[]
        for future in prediction['futures']:
            exact,after = equilibrium_cycle(trace[-1]['maximum_opening_m'],energy,future['drive_m'],law,fixture)
            separated = bool(damage(after,energy,law) >= 1-1e-12)
            future_errors.append(float(((future['mean_gap_m']-exact)*1000)**2))
            coverages.append(future['q025_gap_m']-1e-12 <= exact <= future['q975_gap_m']+1e-12)
            briers.append((future['separation_mass']-separated)**2)
            width = future['q975_gap_m']-future['q025_gap_m'];widths.append(width*1000)
            scores.append(float((width+40*max(future['q025_gap_m']-exact,exact-future['q975_gap_m'],0.))*1000))
        last_cost = decision[-1]['cost']
        prefix_queries = last_cost['equilibrium_cycle_evaluations']
        logical_queries = sum(d['cost']['information_action_candidate_queries']+d['cost']['guard_action_candidate_queries']
            +d['cost']['witness_construction_queries'] for d in decision)
        audit_queries = sum(d['cost']['post_selection_audit_queries'] for d in decision)
        controls = e['policy'] in ('rest_only','fixed_probe')
        rows.append({'episode_id':e['id'],'context_id':e['context_id'],'case_id':p['case_id'],'in_prior':p['in_prior'],
            'policy':e['policy'],'sigma_m':e['sigma_m'],'actual_bias_m':p['actual_bias_m'],
            'true_extra_damage':actual_extra,'true_cap_violation':bool(actual_extra > settings['damage_cap']+1e-12),
            'new_separation_during_observation':bool(trace[-1]['damage'] >= 1-1e-12 and reference < 1-1e-12),
            'true_parameters_compatible_with_observation_bounds':bool(p['in_prior'] and compatible[0]),
            'actual_noise_and_bias_within_assumed_bounds':bool(abs(p['actual_bias_m']) <= settings['bias_bound_m']
                and all(abs(r['noise_m']) <= settings['noise_multiple']*e['sigma_m'] for r in trace)),
            'empty_set_steps':sum(d['empty_set'] for d in decision),
            'selected_action_forbidden_by_full_grid_steps':sum(d['chosen_action_finite_sample_forbidden'] is True for d in decision),
            'optimistic_action_labels':sum(d['optimistic_admissibility_count'] or 0 for d in decision),
            'fine_grid_forbidden_steps':sum(d['fine_selected_action_forbidden'] is True for d in finer),
            'fine_resolved_coarse_empty_steps':sum(d['coarse_empty'] and d['fine_candidates']>0 for d in finer),
            'probe_actions':sum(a > settings['weak_drive_m'] for a in e['actions_m'][2:]),
            'rest_actions':sum(a == 0. for a in e['actions_m'][2:]),
            'witness_union_max_size':max(d['witness_union_size'] for d in decision),
            'mean_future_squared_error_mm2':float(np.mean(future_errors)),
            'empirical_95_coverage':float(np.mean(coverages)),'separation_brier':float(np.mean(briers)),
            'mean_interval_width_mm':float(np.mean(widths)),'mean_interval_score_95_mm':float(np.mean(scores)),
            'planning_cycle_queries':logical_queries+(0 if controls else prefix_queries),
            'diagnostic_cycle_queries':audit_queries+(prefix_queries if controls else 0)})
    groups = []
    for key in sorted({(r['in_prior'],r['sigma_m'],r['actual_bias_m'],r['policy']) for r in rows}):
        subset = [r for r in rows if (r['in_prior'],r['sigma_m'],r['actual_bias_m'],r['policy'])==key]
        groups.append(dict(zip(('in_prior','sigma_m','actual_bias_m','policy'),key),episodes=len(subset),
            gap_rmse_mm=float(np.sqrt(np.mean([r['mean_future_squared_error_mm2'] for r in subset]))),
            cap_violation_fraction=float(np.mean([r['true_cap_violation'] for r in subset])),
            maximum_true_extra_damage=max(r['true_extra_damage'] for r in subset),
            mean_true_extra_damage=float(np.mean([r['true_extra_damage'] for r in subset])),
            mean_95_coverage=float(np.mean([r['empirical_95_coverage'] for r in subset])),
            mean_interval_score_95_mm=float(np.mean([r['mean_interval_score_95_mm'] for r in subset])),
            mean_probe_actions=float(np.mean([r['probe_actions'] for r in subset])),
            empty_set_steps=sum(r['empty_set_steps'] for r in subset),
            full_grid_forbidden_steps=sum(r['selected_action_forbidden_by_full_grid_steps'] for r in subset),
            fine_grid_forbidden_steps=sum(r['fine_grid_forbidden_steps'] for r in subset),
            optimistic_action_labels=sum(r['optimistic_action_labels'] for r in subset),
            noise_or_bias_bound_violation_episodes=sum(not r['actual_noise_and_bias_within_assumed_bounds'] for r in subset),
            continuous_observation_constraints_incompatible_episodes=sum(not r['true_parameters_compatible_with_observation_bounds'] for r in subset),
            true_cap_violations_within_noise_bounds=sum(r['true_cap_violation'] and r['actual_noise_and_bias_within_assumed_bounds'] for r in subset),
            mean_planning_cycle_queries=float(np.mean([r['planning_cycle_queries'] for r in subset]))))
    write_json(root/'evaluation-rows.json',rows)
    assert seals == {name:digest(root/name) for name in seals}
    return groups,rows


def summarize(root,report):
    groups = report['groups']
    lines = ['# 유한 후보의 누적 손상 검사','',
        '현재 안정 준정적 모형에서 무하중+약한 입력 관측 뒤 두 번 행동을 다시 선택한다. 실제 영상·엔진·신경망을 쓰지 않는다. 26개 물리 상태 × 3개 잡음 × 3개 실제 편향 = 234개 조건에 6개 정책을 적용했다. 1,404개 정책 실행을 독립 시편 수로 세지 않는다.','',
        '손상 한도 0.02는 최초 약한 관측 후 D 대비 누적 추가 D다. 매 행동마다 기준을 초기화하지 않는다. 후보마다 준비부터 실제 정책 행동 전체를 재생한다. 정책별 미래 정답도 해당 정책의 손상 이후 상태에서 따로 계산한다.','',
        '## 정책','',
        '- rest_only: 추가 하중 없이 무하중 관측만 두 번 더 한다.',
        '- fixed_probe: 기존 1.65/1.85mm 입력을 그대로 실행한다.',
        '- top8_guard: 관측 집합 안에서 posterior 가중치가 높은 최대 8개만 위험 검사.',
        '- full_guard: 관측 조건을 만족하는 전체 유한 후보를 위험 검사.',
        '- witness_guard: 전체 위험을 먼저 계산해 금지 행동마다 위반 후보 하나를 top8에 추가. 원래 행동 격자의 허용/금지 판정을 보존하지만 전체 탐색 비용이 필요하다.',
        '- full_guard_info8: 위험은 전체 후보, 정보 목적함수는 상위 8개만 사용.','',
        '정보 목적은 공통 편향을 조건부로 반영한 다음 관측 평균의 분산에 기반한 표준 disagreement proxy다. 정확한 mutual information 또는 새 방법이라고 부르지 않는다.','',
        '## 대표 조건: 범위 안 24개 상태, 잡음 0.020mm, 실제 편향 +0.030mm','',
        '| 정책 | 미래 RMSE (mm) | 실제 누적 한도 위반 | 최대 추가 D | 평균 탐색 횟수 / 2 | 계획 cycle 조회 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for policy in report['settings']['policies']:
        r = next(r for r in groups if r['in_prior'] and r['sigma_m']==.00002 and r['actual_bias_m']==.00003 and r['policy']==policy)
        lines.append(f"| {policy} | {r['gap_rmse_mm']:.4f} | {r['cap_violation_fraction']:.1%} | {r['maximum_true_extra_damage']:.5f} | {r['mean_probe_actions']:.2f} | {r['mean_planning_cycle_queries']:.0f} |")
    lines += ['', '## 유한 격자의 누락과 전체 결과','',
        '| 정책 | 범위 안 전체 조건의 실제 한도 위반 실행 | 전체/소수 격자 판정 불일치 수 | 161² 감사의 금지 선택 수 / 432회 |',
        '| --- | ---: | ---: | ---: |']
    for policy in report['settings']['policies']:
        subset = [r for r in groups if r['in_prior'] and r['policy']==policy]
        violations = round(sum(r['cap_violation_fraction']*r['episodes'] for r in subset))
        optimistic = sum(r['optimistic_action_labels'] for r in subset)
        finer = sum(r['fine_grid_forbidden_steps'] for r in subset)
        fine_label = str(finer) if policy not in ('rest_only','fixed_probe') else '검사 대상 아님'
        lines.append(f'| {policy} | {violations} / 216 | {optimistic} | {fine_label} |')
    lines += ['', '판정 불일치 수는 2단계×7개 설계 행동의 허용/금지 라벨 비교다. 실제 실행한 행동 위반 횟수와 다르다. 161² 감사는 4개 guarded 정책의 실제 선택 행동만 대상으로 한다. 더 조밀한 집합에서 금지로 바뀌어도 실제 시편이 한도를 넘었다는 뜻은 아니다.', '',
        '## 보장의 범위','',
        '전체 후보는 연속 상태 전체가 아니라 81×81 사전 격자 중 유계 오차 조건을 만족하는 점들이다. 각 관측 오차 ±3σ, 공통 bias ±0.150mm의 교집합을 사용한다. 생성 잡음은 Gaussian이므로 이 경계를 벗어날 수 있다. Gaussian posterior와 이 유계 집합은 다른 목적의 표현이다.','',
        '이 집합의 최대 손상은 해당 유한 점에 대한 최댓값이다. 실제 참 상태가 격자 사이·사전 범위 밖이거나 관측/물리 모델이 틀리면 실제 한도 위반이 생길 수 있다. ‘전체 검사=실물 안전’이라고 하지 않는다. 빈 집합이면 guarded 정책은 무하중으로 기권하며, 후보를 몰래 다시 넣지 않는다.','',
        '전체 검사 정책의 실제 추가 손상은 이번 범위 안 모든 조건에서 0이었다. 안전성과 유용성을 함께 달성했다고 해석하지 않는다. 대표 조건에서 미래 오차는 rest_only와 비슷해, 의미 있는 예측 이득을 확보했다고 보기 어렵다. full_guard_info8의 적은 논리 조회량은 단순 계산 분리 기준선의 결과이며 실제 실행 시간이나 새로운 방법의 우위가 아니다.','',
        'full_guard와 witness_guard는 같은 조건에서 행동과 관측이 정확히 같음을 확인했다. 이는 단순 위험 증거 기준선이 현재 행동 격자를 보존함을 확인하는 것이며 새로운 알고리즘 효과가 아니다. 관측이 갱신될 때마다 다시 전체 탐색했고 그 비용을 포함했다.','',
        'top8의 전체 격자 감사와 선택한 행동의 161×161 감사는 행동 결정 후 수행한다. 감사 결과를 정책에 되돌려주지 않는다. 계산량에는 계획/사후 감사를 따로 기록했다. 무하중·고정 입력 대조군의 관측 집합 계산은 정책에 필요 없는 사후 진단으로 분리한다.','',
        '실제 noise/bias가 유계 가정을 만족했는지, 실제 파라미터가 연속 관측 제약과 사전 범위에 부합하는지, 유한 후보가 비었는지, 더 조밀한 격자가 위반을 찾았는지를 따로 기록한다. 연속 제약과의 부합도 실제 참 상태가 유한 격자에 포함됐다는 뜻은 아니다.','',
        '현재 같은 재료의 안정 단조 완전 분리 기준은 이력에 따라 바뀌지 않는다. 미래 분리 예측을 재료/이력의 결합 모호성 과제로 해석한다. 실제 안전 지표·일반 3D 파괴·학습 월드모델·신규 방법 기여는 검증하지 않았다.','']
    (root/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [next(r for r in groups if r['in_prior'] and r['sigma_m']==.00002 and r['actual_bias_m']==.00003 and r['policy']==p) for p in report['settings']['policies']]
    fig,axes = plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
    labels = ['rest','fixed','top8','full','witness','full/info8']
    axes[0].bar(labels,[r['gap_rmse_mm'] for r in selected]);axes[0].set(ylabel='Future hold-gap RMSE (mm)',title='Each policy has its own resulting truth')
    axes[1].bar(labels,[100*r['cap_violation_fraction'] for r in selected]);axes[1].set(ylabel='True cumulative cap violations (%)',title='D cap 0.02; finite grid is not a guarantee')
    axes[2].bar(labels,[r['mean_planning_cycle_queries'] for r in selected]);axes[2].set(ylabel='Logical planning cycle queries',title='Witness construction included; audit separate')
    for axis in axes:
        axis.grid(axis='y',alpha=.2);axis.tick_params(axis='x',labelrotation=25)
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)


def main():
    project = Path(os.environ['WM_PROJECT_ROOT'])
    settings = json.loads((project/'config/finite-damage-guard.json').read_text())
    source = Path(json.loads((project/settings['source_report']).read_text())['run_directory'])
    base = json.loads((source/'config.json').read_text());cases = json.loads((source/'private-cases.json').read_text())
    source_hashes = {name:digest(source/name) for name in ('observations.json','private-cases.json','report.json','posteriors.npz','forecasts.json')}
    root = Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-finite-damage-guard')
    root.mkdir(parents=True);print(f'OUTPUT={root}',flush=True)
    write_json(root/'config.json',settings);write_json(root/'physics-config.json',base)
    sources = [Path(__file__),project/'wmfracture/damage_guard.py',project/'wmfracture/compliant_inference.py',
        project/'wmfracture/measurement_bias.py',project/'wmfracture/cohesive.py',project/'wmfracture/compliant_fixture.py']
    for path in sources:
        target = root/'source'/path.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in sources})
    law = InterfaceLaw(**json.loads((project/base['law_config']).read_text())['law']);fixture = Fixture(**base['fixture'])
    shutil.copy2(project/base['law_config'],root/'law-config.json')
    started = time.perf_counter();collect(root,settings,cases,law,fixture)
    collection_seconds = time.perf_counter()-started
    seals = forecast(root,settings,base,law,fixture)
    fine_grid_audit(root,settings,law,fixture)
    groups,rows = evaluate(root,settings,law,fixture,seals)
    assert source_hashes == {name:digest(source/name) for name in source_hashes}
    report = {'status':'finite_candidate_damage_guard_completed','run_directory':str(root),'source_run':str(source),
        'settings':settings,'physical_cases':len(cases),'policy_episodes':len(rows),'groups':groups,
        'collection_seconds':collection_seconds,'total_seconds':time.perf_counter()-started,
        'source_files_unchanged':True,'source_sha256':source_hashes,'seals':seals,
        'full_and_witness_actions_and_observations_identical':True,
        'continuous_safety_bound_validated':False,'new_method_novelty_validated':False,'real_data_used':False}
    write_json(root/'report.json',report);summarize(root,report)
    out = project/'artifacts/finite-damage-guard';out.mkdir(parents=True,exist_ok=True)
    for name in ('report.json','summary.md','comparison.png','config.json','code-sha256.json','decisions.json','fine-grid-audit.json','evaluation-rows.json','sealed-predictions.json','forecast-cost.json'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','settings','seals','source_sha256')},indent=2),flush=True)


if __name__ == '__main__':
    main()
