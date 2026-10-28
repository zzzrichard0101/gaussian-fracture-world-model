"""Paired shared-offset diagnostic on immutable synthetic-filter observations."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest, write_json
from synthetic_filter_benchmark import forecast, truth_prefix
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import PrefixModel, grid, normalize, ParticleFilter, equilibrium_cycle, damage
from wmfracture.measurement_bias import GaussianObservation


def prepare(root, source, settings, base):
    episodes = json.loads((source/'observations.json').read_text())
    # Inference input deliberately excludes protocol/scenario/case labels.
    public = [{key:e[key] for key in ('id','sigma_m','actions_m','observed_gaps_m')} for e in episodes]
    if settings['additional_synthetic_zero_load_calibration']:
        for episode, observed in zip(episodes, public):
            # NEW synthetic observation, not present in the original recordings.
            # True bias is used only here in the data generator, never in fit().
            sigma_index = base['gap_noise_sigma_m'].index(episode['sigma_m'])
            case_index = int(episode['case_id'].split('-')[-1])
            rng = np.random.default_rng(settings['seed']+900000+100*case_index+sigma_index)
            observed['rest_gap_m'] = float(base['observation_scenarios'][episode['scenario']]
                +rng.normal(0,episode['sigma_m']))
    metadata = [{key:e[key] for key in ('id','case_id','protocol','scenario')} for e in episodes]
    write_json(root/'observations.json', public)
    write_json(root/'evaluation-episodes.json', metadata)
    shutil.copy2(source/'private-cases.json', root/'private-cases.json')
    return public


def fit(root, settings, base, law, fixture):
    # Only public scalar observations and prior/settings; no evaluation labels.
    episodes = json.loads((root/'observations.json').read_text())
    model = PrefixModel(law, fixture, base['parameter_bounds'])
    unit = grid(settings['grid_side']); archive = {'grid':unit}; packets = []
    bank = {}
    for e in episodes:
        actions = tuple(e['actions_m'])
        if actions not in bank:
            bank[actions] = model.predict(unit, actions)[0]
    library_cost = dict(model.cost)
    for index, e in enumerate(episodes):
        actions, y, sigma = e['actions_m'], np.asarray(e['observed_gaps_m']), e['sigma_m']
        variants = [(mode,False) for mode in settings['observation_modes']]
        if settings['additional_synthetic_zero_load_calibration']:
            variants.append(('shared',True))
        for mode, rest in variants:
            input_actions = ([0.]+actions) if rest else actions
            input_y = np.r_[e['rest_gap_m'],y] if rest else y
            suffix = '_rest' if rest else ''
            likelihood = GaussianObservation(mode, settings['bias_sigma_m'])
            for stage in range(1,len(y)+1):
                predictions = bank[tuple(actions)][:,:stage]
                if rest:
                    predictions = np.column_stack((np.zeros(len(unit)),predictions))
                observed = input_y[:stage+int(rest)]
                weights = normalize(likelihood(predictions,observed,sigma))
                key = f'packet{len(packets)}'; archive[key+'_weights'] = weights
                packet = {'episode_id':e['id'], 'stage':stage, 'method':f'grid{settings["grid_side"]}_{mode}{suffix}',
                    'unit_key':'grid','weight_key':key+'_weights','observation_mode':mode,
                    'rest_calibration_used':rest,'observations_used':stage+int(rest),
                    'cost':{'parameter_prefix_evaluations':len(unit), 'equilibrium_cycle_evaluations':len(unit)*(stage+1+int(rest))}}
                if mode == 'shared':
                    packet['bias_posterior'] = likelihood.bias_moments(predictions,observed,sigma,weights)
                packets.append(packet)
            for replicate in range(base['smc']['replicates']):
                model.reset_cost()
                seed = settings['seed']+100000+index*100+replicate
                smc = ParticleFilter(model,base['smc']['particles'],seed,base['smc']['move_steps'],
                    base['smc']['move_scale'],base['smc']['global_probability'],likelihood=likelihood)
                for stage in range(1,len(y)+1):
                    trace = smc.update(input_actions[:stage+int(rest)],input_y[:stage+int(rest)],sigma)
                    key = f'packet{len(packets)}'
                    archive[key+'_unit'], archive[key+'_weights'] = smc.unit.copy(), smc.weights.copy()
                    packet = {'episode_id':e['id'],'stage':stage,'method':f'smc128_{mode}{suffix}','replicate':replicate,
                        'unit_key':key+'_unit','weight_key':key+'_weights','observation_mode':mode,
                        'rest_calibration_used':rest,'observations_used':stage+int(rest),
                        'trace':trace,'cost':dict(model.cost)}
                    if mode == 'shared':
                        packet['bias_posterior'] = likelihood.bias_moments(smc.predictions,input_y[:stage+int(rest)],sigma,smc.weights)
                    packets.append(packet)
        # Predeclared sensitivity; no truth used to select a winning width.
        for tau in settings['sensitivity_bias_sigma_m']:
            if tau == settings['bias_sigma_m']:
                continue
            likelihood = GaussianObservation('shared',tau)
            predictions = bank[tuple(actions)]
            weights = normalize(likelihood(predictions,y,sigma))
            key = f'packet{len(packets)}'; archive[key+'_weights'] = weights
            packets.append({'episode_id':e['id'],'stage':len(y),'method':f'grid{settings["grid_side"]}_shared_tau{tau*1000:g}mm',
                'unit_key':'grid','weight_key':key+'_weights','observation_mode':'shared','bias_sigma_m':tau,
                'bias_posterior':likelihood.bias_moments(predictions,y,sigma,weights),
                'cost':{'parameter_prefix_evaluations':len(unit),'equilibrium_cycle_evaluations':len(unit)*(len(y)+1)}})
        if index % 80 == 0:
            print(f'BIAS_FIT {index+1}/{len(episodes)}',flush=True)
    np.savez_compressed(root/'posteriors.npz',**archive)
    write_json(root/'fit-packets.json',packets)
    write_json(root/'library-cost.json',library_cost)


def evaluate(root, base, law, fixture, seals):
    # Called only after posterior and physical forecasts are sealed.
    episodes = {e['id']:e for e in json.loads((root/'observations.json').read_text())}
    metadata = {e['id']:e for e in json.loads((root/'evaluation-episodes.json').read_text())}
    cases = {c['id']:c for c in json.loads((root/'private-cases.json').read_text())}
    predictions = json.loads((root/'forecasts.json').read_text())
    rows, bias_rows = [], []
    for prediction in predictions:
        episode, meta = episodes[prediction['episode_id']], metadata[prediction['episode_id']]
        case = cases[meta['case_id']]; energy = case['parameters'][0]
        _, maxima = truth_prefix(case['parameters'],episode['actions_m'][:prediction['stage']],law,fixture)
        header = {'episode_id':episode['id'],'case_id':case['id'],'in_prior':case['in_prior'],
            'protocol':meta['protocol'],'scenario':meta['scenario'],'sigma_m':episode['sigma_m'],
            'stage':prediction['stage'],'method':prediction['method'],'replicate':prediction.get('replicate')}
        if 'bias_posterior' in prediction:
            truth_bias = base['observation_scenarios'][meta['scenario']]
            bias_rows.append({**header,**prediction['bias_posterior'],'true_bias_m':truth_bias,
                'error_mm':(prediction['bias_posterior']['mean_m']-truth_bias)*1000})
        for future in prediction['futures']:
            exact, maximum = equilibrium_cycle(maxima[-1],energy,future['drive_m'],law,fixture)
            separated = bool(damage(maximum,energy,law) >= 1-1e-12)
            rows.append({**header,'drive_m':future['drive_m'],
                'squared_error_mm2':float(((future['mean_gap_m']-exact)*1000)**2),
                'covered_95':bool(future['q025_gap_m']-1e-12 <= exact <= future['q975_gap_m']+1e-12),
                'interval_width_mm':(future['q975_gap_m']-future['q025_gap_m'])*1000,
                'interval_score_95_mm':float((future['q975_gap_m']-future['q025_gap_m']
                    +40*max(future['q025_gap_m']-exact,exact-future['q975_gap_m'],0.))*1000),
                'brier':float((future['separation_mass']-separated)**2)})
    write_json(root/'evaluation-rows.json',rows); write_json(root/'bias-evaluation.json',bias_rows)
    groups = []
    keys = sorted({(r['in_prior'],r['protocol'],r['scenario'],r['sigma_m'],r['stage'],r['method']) for r in rows})
    for key in keys:
        subset = [r for r in rows if (r['in_prior'],r['protocol'],r['scenario'],r['sigma_m'],r['stage'],r['method']) == key]
        bias = [r for r in bias_rows if (r['in_prior'],r['protocol'],r['scenario'],r['sigma_m'],r['stage'],r['method']) == key]
        groups.append(dict(zip(('in_prior','protocol','scenario','sigma_m','stage','method'),key),
            gap_rmse_mm=float(np.sqrt(np.mean([r['squared_error_mm2'] for r in subset]))),
            empirical_95_coverage=float(np.mean([r['covered_95'] for r in subset])),
            separation_brier=float(np.mean([r['brier'] for r in subset])),
            mean_interval_width_mm=float(np.mean([r['interval_width_mm'] for r in subset])),
            mean_interval_score_95_mm=float(np.mean([r['interval_score_95_mm'] for r in subset])),
            bias_rmse_mm=float(np.sqrt(np.mean([r['error_mm']**2 for r in bias]))) if bias else None))
    assert seals == {'posteriors':digest(root/'posteriors.npz'),'packets':digest(root/'fit-packets.json'),'forecasts':digest(root/'forecasts.json')}
    return groups


def refine(root, settings, base, law, fixture):
    # Empirical grid check: no labels, original fits unchanged, final stage only.
    episodes = json.loads((root/'observations.json').read_text())
    original = {(r['episode_id'],r['method']):r for r in json.loads((root/'forecasts.json').read_text())
        if r['stage']==3 and r['method'].startswith('grid') and 'tau' not in r['method']}
    model = PrefixModel(law,fixture,base['parameter_bounds'])
    unit = grid(settings['refinement_side']); energy = model.parameters(unit)[:,0]
    records = []
    for actions in {tuple(e['actions_m']) for e in episodes}:
        predictions, maximum = model.predict(unit,actions)
        future_values = []
        for command in base['future_drives_m']:
            gap, after = equilibrium_cycle(maximum,energy,command,law,fixture)
            future_values.append((gap,damage(after,energy,law) >= 1-1e-12))
        for e in episodes:
            if tuple(e['actions_m']) != actions:
                continue
            for mode in settings['observation_modes']:
                weights = normalize(GaussianObservation(mode,settings['bias_sigma_m'])(predictions,e['observed_gaps_m'],e['sigma_m']))
                old = original[(e['id'],f'grid{settings["grid_side"]}_{mode}')]
                gap_diff, event_diff = [], []
                for (gap,event),old_future in zip(future_values,old['futures']):
                    gap_diff.append(abs(np.sum(weights*gap)-old_future['mean_gap_m'])*1000)
                    event_diff.append(abs(np.sum(weights*event)-old_future['separation_mass']))
                records.append({'episode_id':e['id'],'mode':mode,'maximum_mean_gap_difference_mm':float(max(gap_diff)),
                    'maximum_event_mass_difference':float(max(event_diff))})
    write_json(root/'grid-refinement.json',{'sides':[settings['grid_side'],settings['refinement_side']],
        'records':records,'scope':'Final-stage quadrature diagnostic, not a convergence proof.'})


def summarize(root, report):
    lines = ['# 공통 측정 편향을 고려한 추론 기준선','',
        '동일한 이전 관측을 유지하고 오차 가정만 바꾼 개발 진단이다. 참 편향·물성을 추정기에 제공하지 않는다. 26개 물리 사례·312개 관측 조건이며 독립 시편 312개가 아니다.','',
        '- independent: 기존 독립 잡음만 가정.',
        '- inflated: 매 관측의 잡음 폭을 sqrt(sigma²+tau²)로 확대. 공통 편향의 시간 상관은 무시.',
        '- shared: 관측 구간 전체에서 같은 b를 사용. b~N(0,tau²)를 적분해 공분산 sigma²I+tau²11ᵀ를 사용.', '',
        '- shared_rest: 실제 하중 관측 3회에 더해, 무하중 틈이 0이라는 모형의 기준점에서 합성 관측 1회를 추가. 추가 손상은 없지만 관측 예산은 1회 더 쓴다.', '',
        '주 결과는 tau=0.050mm로 고정했다. 0.010/0.100mm 민감도 결과도 별도 방법으로 남겼고, 정답으로 최적 폭을 선택하지 않았다. 같은 물리 사전분포·하중·SMC 예산을 사용한다.','',
        '## 범위 안 24개 사례, 관측 sigma=0.020mm, 고정 탐색 입력, 관측 3회','',
        '| 실제 조건 | 방법 | 미래 RMSE (mm) | 명목 95% 포함률 | 평균 구간 폭 (mm) | 구간 점수 ↓ (mm) | 분리 Brier | 편향 RMSE (mm) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for scenario in ('matched','unmodelled_constant_bias'):
        for method in ('grid81_independent','grid81_inflated','grid81_shared','grid81_shared_rest',
            'smc128_independent','smc128_inflated','smc128_shared','smc128_shared_rest'):
            r = next(r for r in report['groups'] if r['in_prior'] and r['protocol']=='fixed_probe' and r['scenario']==scenario
                and r['sigma_m']==.00002 and r['stage']==3 and r['method']==method)
            bias = '—' if r['bias_rmse_mm'] is None else f"{r['bias_rmse_mm']:.4f}"
            label = '실제 편향 없음' if scenario=='matched' else '+0.030mm 편향'
            lines.append(f"| {label} | {method} | {r['gap_rmse_mm']:.4f} | {r['empirical_95_coverage']:.1%} | {r['mean_interval_width_mm']:.4f} | {r['mean_interval_score_95_mm']:.4f} | {r['separation_brier']:.4f} | {bias} |")
    lines += ['', '## 해석과 한계','',
        '공통 편향 추정은 정확한 Gc·손상 이력·편향의 유일 식별을 보장하지 않는다. 약한 입력만 반복하면 물리 변화와 offset을 분리하지 못할 수 있다. 편향 posterior는 여러 물리 가설을 섞은 분포라 평균·표준편차만 저장하고 이를 Gaussian 95% 구간이라고 부르지 않는다.','',
        '95% 구간 점수는 폭 + 40×구간 밖 정답까지의 거리다. 작을수록 좋으며, 넓히기만 한 구간과 과도하게 좁은 구간을 모두 벌점화한다. 경험 포함률이 100%라는 사실만으로 보정됐다고 하지 않는다.','',
        'rest 관측은 원래 데이터에 없던 추가 합성 값이다. 무하중 출력=0·동일 offset 유지·독립 측정 잡음을 가정한다. 실제 장치에서 기준점이 불명확하거나 편향이 시간에 따라 변하면 이 보정이 성립하지 않는다. 이 단순 보정을 먼저 비교하지 않고 능동 개입의 필요성을 주장하지 않는다.','',
        '실제 편향 0인 조건에서는 공유 편향을 허용한 만큼 불확실성이 커지거나 점 예측이 나빠질 수 있다. 각 조건·잡음·프로토콜을 나눠 읽어야 한다. 이 데이터의 조건은 알려진 고정 bias이므로 사전분포에서 bias를 무작위 추출한 보정 시험과도 다르다.','',
        '기존 고정 탐색 하중은 그대로여서 손상을 줄인 실험이 아니다. 모델링한 것은 한 구간 내 일정 offset뿐이며 scale·drift·추적 실패·물리 법칙 불일치는 다루지 않는다. 표준 Gaussian nuisance 추론/SMC로 새로운 방법이나 논문 기여를 주장하지 않는다.','',
        '조밀한 격자도 수렴 정답이 아니며, 81²→321² 최종 단계 민감도는 grid-refinement.json에 별도로 기록했다. 원래 실험의 결과/관측은 변경하지 않았다.','']
    (root/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(15,4.1),constrained_layout=True)
    modes = ('independent','inflated','shared','shared_rest')
    labels = ('independent','inflated','shared','shared + rest')
    for scenario, color in (('matched','tab:blue'),('unmodelled_constant_bias','tab:orange')):
        selected = [next(r for r in report['groups'] if r['in_prior'] and r['protocol']=='fixed_probe'
            and r['scenario']==scenario and r['sigma_m']==.00002 and r['stage']==3 and r['method']==f'smc128_{mode}') for mode in modes]
        label = 'True bias 0 mm' if scenario=='matched' else 'True bias +0.030 mm'
        axes[0].plot(labels,[r['gap_rmse_mm'] for r in selected],'o-',label=label,color=color)
        axes[1].plot(labels,[r['empirical_95_coverage'] for r in selected],'o-',label=label,color=color)
        axes[2].plot(labels,[r['mean_interval_score_95_mm'] for r in selected],'o-',label=label,color=color)
    axes[0].set(ylabel='Future hold-gap RMSE (mm)',title='Rest adds one observation; same 128 particles')
    axes[1].set(ylabel='Empirical nominal-95% coverage',ylim=(0,1.03),title='No calibration guarantee')
    axes[2].set(ylabel='95% interval score (mm; lower is better)',title='Penalizes interval width and misses')
    for axis in axes:
        axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--source-run',type=Path);args = parser.parse_args()
    project = Path(os.environ['WM_PROJECT_ROOT'])
    settings = json.loads((project/'config/bias-aware-filter.json').read_text())
    source = args.source_run or Path(json.loads((project/settings['source_report']).read_text())['run_directory'])
    base = json.loads((source/'config.json').read_text())
    original_hashes = {name:digest(source/name) for name in ('observations.json','private-cases.json','posteriors.npz','fit-packets.json','forecasts.json','report.json')}
    root = Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-bias-aware-filter')
    root.mkdir(parents=True);print(f'OUTPUT={root}',flush=True)
    write_json(root/'config.json',settings);write_json(root/'physics-config.json',base)
    write_json(root/'source-manifest.json',{'source_run':str(source),'sha256':original_hashes})
    sources = [Path(__file__),project/'wmfracture/measurement_bias.py',project/'wmfracture/compliant_inference.py',
        project/'wmfracture/cohesive.py',project/'wmfracture/compliant_fixture.py',
        project/'scripts/synthetic_filter_benchmark.py',project/'scripts/history_refinement.py']
    for path in sources:
        target = root/'source'/path.relative_to(project)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in sources})
    law = InterfaceLaw(**json.loads((project/base['law_config']).read_text())['law']);fixture = Fixture(**base['fixture'])
    shutil.copy2(project/base['law_config'],root/'law-config.json')
    prepare(root,source,settings,base);started = time.perf_counter();fit(root,settings,base,law,fixture)
    fit_seconds = time.perf_counter()-started
    seals = forecast(root,base,law,fixture);write_json(root/'sealed-predictions.json',seals)
    groups = evaluate(root,base,law,fixture,seals);refine(root,settings,base,law,fixture)
    assert original_hashes == {name:digest(source/name) for name in original_hashes}
    report = {'status':'bias_aware_scalar_filter_completed','run_directory':str(root),'source_run':str(source),
        'groups':groups,'fit_seconds':fit_seconds,'total_seconds':time.perf_counter()-started,
        'source_files_unchanged':True,'source_sha256':original_hashes,'seals':seals,
        'novel_method_validated':False,'risk_control_implemented':False,'real_data_used':False,
        'scope':settings['scope']}
    write_json(root/'report.json',report);summarize(root,report)
    out = project/'artifacts/bias-aware-filter';out.mkdir(parents=True,exist_ok=True)
    for name in ('report.json','summary.md','comparison.png','config.json','code-sha256.json','source-manifest.json','grid-refinement.json','bias-evaluation.json','evaluation-rows.json'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','source_sha256','seals')},indent=2),flush=True)


if __name__ == '__main__':
    main()
