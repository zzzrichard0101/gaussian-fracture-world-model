"""Auditable continuous-region risk prototype and counterexample action filter."""
from dataclasses import asdict
from datetime import datetime,timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest,write_json
from finite_damage_guard import SyntheticSpecimen
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import PrefixModel,grid
from wmfracture.damage_guard import FiniteGuard,feasible_bias_interval,action_risks
from wmfracture.continuous_risk import ContinuousRisk,replay_proof


def public_inputs(prefix,sigma,config,law,fixture,reference_count):
    return {'model_kind':'stable_quasistatic_nonnegative_cycles','parameter_bounds':config['parameter_bounds'],'spring':fixture.spring,
        'law':{k:asdict(law)[k] for k in ('area','tensile_stiffness','peak_traction')},
        'actions_m':prefix['actions_m'],'observed_gaps_m':prefix['observed_gaps_m'],
        'reference_count':reference_count,'noise_radius_m':float(config['noise_multiple']*sigma),
        'bias_bound_m':config['bias_bound_m'],'damage_cap':config['damage_cap']}


class WarmBank:
    """Public finite samples propose witnesses only; never supply an upper bound."""
    def __init__(self,inputs,side,law,fixture):
        self.inputs,self.model = inputs,PrefixModel(law,fixture,inputs['parameter_bounds'])
        self.law,self.fixture = law,fixture
        self.unit = grid(side);self.parameters = self.model.parameters(self.unit)
        prediction,self.maximum = self.model.predict(self.unit,inputs['actions_m'])
        _,self.baseline = self.model.predict(self.unit,inputs['actions_m'][:inputs['reference_count']])
        self.mask,_,_ = feasible_bias_interval(prediction,inputs['observed_gaps_m'],inputs['noise_radius_m'],1.,inputs['bias_bound_m'])
        self.action_queries = 0

    def points(self,action,count):
        selected = np.flatnonzero(self.mask)
        if not len(selected):
            return []
        risk,_ = action_risks(self.maximum[selected],self.baseline[selected],self.parameters[selected,0],
            [action],self.law,self.fixture)
        self.action_queries += len(selected)
        order = np.argsort(-risk[0],kind='stable')[:count]
        return self.parameters[selected[order]].tolist()


class Queries:
    def __init__(self,root,settings,config,law,fixture):
        self.root,self.settings,self.config,self.law,self.fixture = root,settings,config,law,fixture
        self.rows=[];self.cache={}

    def query(self,label,prefix,sigma,action,budget=None):
        inputs=public_inputs(prefix,sigma,self.config,self.law,self.fixture,self.settings['reference_count'])
        key=json.dumps(inputs,sort_keys=True)
        if key not in self.cache:
            self.cache[key]=(ContinuousRisk(inputs),WarmBank(inputs,self.settings['warm_grid_side'],self.law,self.fixture))
        verifier,bank=self.cache[key]
        warm=bank.points(action,self.settings['warm_point_count'])
        started=time.perf_counter()
        result=verifier.verify(action,max_cells=budget or self.settings['cell_budget'],warm_points=warm)
        solve=time.perf_counter()-started
        started=time.perf_counter();assert replay_proof(result,expected_inputs=inputs,expected_action=action);replay=time.perf_counter()-started
        relative=f'proofs/query-{len(self.rows):03d}.json';path=self.root/relative;path.parent.mkdir(exist_ok=True)
        write_json(path,result)
        row={'label':label,'action_m':action,'decision':result['decision'],
            'cell_budget':budget or self.settings['cell_budget'],'warm_proposed_points':len(warm),
            'lower_damage':result['lower_damage'],'upper_damage':result['upper_damage'],
            'cells':result['evaluated_cells'],'open_leaves':result['open_leaves'],
            'point_evaluations':result['point_evaluations'],'solve_seconds':solve,'proof_replay_seconds':replay,
            'proof_file':relative,'proof_sha256':digest(path),'proof_replayed':True}
        self.rows.append(row)
        print(f"RISK {label} {result['decision']} L={result['lower_damage']} U={result['upper_damage']:.6g} cells={result['evaluated_cells']}",flush=True)
        return row

    def filter_action(self,label,prefix,sigma,proposal):
        first=self.query(label+'-proposal',prefix,sigma,proposal)
        if first['decision']=='allow':
            return proposal,first
        lower=self.query(label+'-zero',prefix,sigma,0.)
        if lower['decision']!='allow':
            return None,lower
        low,high=0.,proposal
        for step in range(self.settings['backtracking_steps']):
            middle=(low+high)/2
            checked=self.query(f'{label}-bisect-{step}',prefix,sigma,middle)
            if checked['decision']=='allow':
                low,lower=middle,checked
            else:
                # Unknown is a computational abstention, not proof of unsafety.
                high=middle
        return low,lower


def audit_recorded(queries,decisions,settings):
    regression=[d for d in decisions if d['episode_id']==settings['regression_episode']]
    if len(regression)!=2:
        raise ValueError('Expected the two recorded regression prefixes')
    for decision in regression:
        prefix,sigma=decision['public_prefix'],decision['sigma_m']
        label=f"recorded-counterexample-round-{decision['round']}"
        queries.query(label+'-small-budget',prefix,sigma,decision['action_m'],settings['small_budget'])
        queries.query(label,prefix,sigma,decision['action_m'])
    first=regression[0]
    queries.query('initial-rest-control',first['public_prefix'],first['sigma_m'],0.)
    queries.query('initial-weak-control',first['public_prefix'],first['sigma_m'],queries.config['weak_drive_m'])
    # Deterministic public-prefix controls; no private outcome used for selection.
    alternatives=[d for d in decisions if d['policy']=='continuous_guard' and d['round']==0 and d['episode_id']!=settings['regression_episode']]
    chosen=[];seen=set()
    for d in alternatives:
        context=(d['sigma_m'],d['episode_id'].split('-')[1])
        if context not in seen:
            seen.add(context);chosen.append(d)
        if len(chosen)==settings['additional_initial_contexts']:
            break
    for index,d in enumerate(chosen):
        queries.query(f'public-prefix-control-{index}',d['public_prefix'],d['sigma_m'],d['action_m'])
    return first


def paired_filter_replay(root,queries,settings,source_config,private,law,fixture,sigma,expected_initial):
    """Private mechanics generate observations; controller/verifier see none of it."""
    case_index=int(private['case_id'].split('-')[-1])
    bias=private['actual_bias_m']
    bias_index=source_config['actual_bias_m'].index(bias)
    seed=source_config['seed']+case_index*1000+bias_index*100
    specimen=SyntheticSpecimen(private['parameters'],bias,sigma,seed,law,fixture)
    controller=FiniteGuard(law,fixture,queries.config,sigma)
    actions=[0.,queries.config['weak_drive_m']];observed=[specimen.observe(a) for a in actions]
    if actions!=expected_initial['actions_m'] or not np.array_equal(observed,expected_initial['observed_gaps_m']):
        raise ValueError('Paired generator failed to reproduce the original initial observations')
    executed=[]
    for turn in range(2):
        prefix={'actions_m':actions.copy(),'observed_gaps_m':observed.copy()}
        proposal,_=controller.plan(actions,observed,'continuous_guard',turn)
        selected,proof=queries.filter_action(f'filtered-round-{turn}',prefix,sigma,proposal)
        if selected is None:
            executed.append({'round':turn,'status':'abstained_without_executing','proof_file':proof['proof_file']})
            break
        actions.append(selected);observed.append(specimen.observe(selected))
        executed.append({'round':turn,'status':'executed_after_allow','proposal_m':proposal,
            'selected_m':selected,'proof_file':proof['proof_file']})
    extra=specimen.trace[-1]['damage']-specimen.trace[1]['damage']
    write_json(root/'filtered-public-trajectory.json',{'actions_m':actions,'observed_gaps_m':observed,'sigma_m':sigma,'decisions':executed})
    write_json(root/'filtered-private-trajectory.json',{'parameters':private['parameters'],'actual_bias_m':bias,'trace':specimen.trace})
    return {'rounds':executed,'new_policy_extra_damage':extra,'old_policy_extra_damage':private['trace'][-1]['damage']-private['trace'][1]['damage'],
        'paired_initial_observations_exactly_reproduced':True,'paired_single_counterexample_only':True,
        'scope':'Conditional quasistatic regression replay; not a new held-out performance result or inertial safety certificate.'}


def summarize(root,report,source_private):
    lines=['# 연속 파라미터 영역의 조건부 위험 검증','',
        '현재 안정 준정적 단일 접합면 모형에서 E=Gc와 관측하지 못한 준비 구동량 H의 연속 사각 영역을 다룬다. 점 몇 개의 최댓값을 상한으로 사용하지 않는다. Gaussian 잡음 대신 선언한 유계 오차·고정 bias 가정에 조건부인 계산이다.','',
        '## 기존 반례 입력 검사','',
        '| 질의 | 예산 | 결과 | 검증된 위험 하한 | 영역 전체 위험 상한 | 검사한 영역 수 |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    for row in report['queries'][:6]:
        lower='미확인' if row['lower_damage'] is None else f"{row['lower_damage']:.6f}"
        lines.append(f"| {row['label']} | {row['cell_budget']} | {row['decision']} | {lower} | {row['upper_damage']:.6f} | {row['cells']} |")
    lines+=['','하한은 모든 관측을 같은 물성·이력·bias로 만족시키는 실제 매개변수 점의 위험이다. 첫 입력의 실제 시편 손상은 0.0103이었지만 다른 가능한 상태에서 0.0417 이상의 손상을 보였으므로 최악 조건 기준으로 거부했다. 실패 검색에서 반례를 못 찾았다는 사실은 허용 근거가 아니다.','',
        'allow는 남은 전체 영역의 상한이 0.02 이하이며 새 분리도 배제하고, 관측에 맞는 점 하나도 확인했을 때만 반환한다. reject는 관측에 맞는 위반 점이 있을 때 반환한다. abstain은 예산 또는 지지집합 확인 부족이다. inconsistent는 전체 영역을 관측과 모순된다고 제외했을 때이며 안전 허용으로 바꾸지 않는다.','',
        '## 반례의 두 단계 입력을 실제로 걸러낸 회귀 실행','',
        '| 단계 | 원래 정책의 새 제안 (mm) | 검증 후 실행 (mm) | 허용 증거의 전체 영역 위험 상한 |',
        '| --- | ---: | ---: | ---: |']
    for row in report['filtered_replay']['rounds']:
        if row['status']=='executed_after_allow':
            proof=json.loads((root/row['proof_file']).read_text())
            lines.append(f"| {row['round']+1} | {row['proposal_m']*1000:.6f} | {row['selected_m']*1000:.6f} | {proof['upper_damage']:.6f} |")
    lines+=['',f"같은 시편 매개변수와 난수를 사용한 새 준정적 궤적의 실제 추가 D는 {report['filtered_replay']['new_policy_extra_damage']:.6f}, 원래 정책은 {report['filtered_replay']['old_policy_extra_damage']:.6f}였다. 새 관측은 선택한 새 입력에서 다시 생성했으며 예전 행동의 관측을 재사용하지 않았다.",'',
        '여기서 추가 손상 0은 정보를 잘 얻었다는 뜻이 아니다. 원래 실패한 한 사례를 막은 회귀 검증이며, 새로운 시험 집합의 예측·수행률 개선은 평가하지 않았다. 보수적인 bisection은 검사한 허용 입력을 선택할 뿐 최대 안전 입력이나 최적 정책을 보장하지 않는다. 판단 보류도 검색 상한을 줄이는 데 사용하며 이를 위험 증거라고 부르지 않는다.','',
        '## 영역을 빠뜨리지 않는 계산','',
        '안정적인 비음수 완전 하중/제하 주기에서는 현재 손상은 지금까지의 최대 구동량 R=max(H,관측 행동들)로 표현된다. D는 R에 따라 증가하고 E에 따라 감소한다. 따라서 각 관측 벌어짐도 H에 따라 증가하고 E에 따라 감소해 사각 영역의 대각 끝점으로 모든 관측을 감쌀 수 있다.','',
        '관측 y_i에 대해 공통 bias b가 [y_i-gap_max-radius, y_i-gap_min+radius]와 선언한 bias 범위 모두에 들어야 한다. 이 교집합이 비면 그 영역은 전부 제외할 수 있다. 남겨진 영역은 실제로 가능한 점을 포함할 수 있는 바깥 근사이며 임의로 feasible이라고 단정하지 않는다.','',
        '누적 손상은 동일한 준비량 H에서 기준 경로와 새 경로를 비교한다. 이 차이는 H에 대해 감소한다. 두 경로 모두에 H의 아래 끝점을 사용한 뒤 E 의존성은 이후 손상의 최댓값−기준 손상의 최솟값으로 감싼다. 영역을 쪼개면 이 차이가 좁아진다. 접착 시작·완전 분리의 분기는 정확한 비교로 처리하며 매 행동마다 손상 기준을 재설정하지 않는다.','',
        '계산은 입력으로 주어진 유한 숫자를 정확한 유리수로 해석하고 사칙연산·비교를 수행한다. float 보고값만 바깥 방향으로 반올림한다. 영역 경계와 이분할 증거는 유리수 문자열로 저장해 직렬화 반올림으로 틈이 생기지 않게 했다.','',
        '## 증거 재검사와 비용','',
        '각 결과는 관측 입력·실제 검사한 입력·지지점/bias·전체 분할 트리와 SHA를 포함한다. 재검사기는 분할이 원래 영역 전체를 덮는지, 제외가 타당한지, 허용 leaf의 상한과 분리 배제가 맞는지, 지지점이 실제 관측을 만족하는지 다시 계산한다. 사용처의 예상 관측·행동과도 대조한다. 정리 증명기로 구현의 수학까지 형식 검증한 것은 아니다.','',
        '유한 161² 후보 bank는 지지점과 위험 하한 후보 제안에만 쓴다. 모든 제안점은 유리수 계산으로 다시 검사한다. 그 bank가 빠뜨린 상태도 영역 상한에서는 제거되지 않는다. 유한 bank 구성·지지점 질의 비용은 warm-costs.json, 영역/점 수와 solve/replay 시간은 queries.json에 저장했다. 이 프로토타입으로 속도 우위나 새 알고리즘 기여를 주장하지 않는다.','',
        '## 적용 범위와 다음 평가','',
        '상한은 알려진 안정 준정적 스칼라 법칙·알려진 구동 입력·선언한 재료/준비 범위·유계 관측 오차에 조건부다. 임의 관성 운동, 시간에 따라 바뀌는 bias, 미지 물리 법칙, Gaussian 꼬리, 실제 물체를 보장하지 않는다. 이전 동적 재생에서 위반이 확인됐다는 사실도 새 허용 증거를 동역학 전체의 증명으로 만들지 않는다.','',
        '다음에는 새로운 상태·잡음에서 full-grid / 위험 증거 / 연속 영역 검증을 같은 조건으로 비교하고, 추가 손상뿐 아니라 예측 오차·수행/기권·전체 계산 비용을 평가한다. 이번에는 기존 반례와 공개된 초기 관측 8개의 회귀 진단을 했으므로 미사용 성능 시험으로 부르지 않는다.','']
    (root/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Patch
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
    reject=next(q for q in report['queries'] if q['label']=='recorded-counterexample-round-0')
    allowed=report['filtered_replay']['rounds'][0]
    proofs=[json.loads((root/reject['proof_file']).read_text()),json.loads((root/allowed['proof_file']).read_text())]
    colors={'infeasible':'#d9d9d9','safe':'#68b984','open':'#f0ba64'}
    center=float(Fraction(proofs[0]['witness']['parameters'][1]))*1000
    for axis,proof,title in zip(axes[:2],proofs,('Original action: feasible violating witness','Filtered action: complete safe/pruned cover')):
        for node in proof['proof_nodes']:
            if node['kind']=='split':
                continue
            box=[[float(Fraction(x)) for x in pair] for pair in node['box']]
            axis.add_patch(Rectangle((box[0][0],box[1][0]*1000),box[0][1]-box[0][0],(box[1][1]-box[1][0])*1000,
                facecolor=colors[node['kind']],edgecolor='white',linewidth=.15))
        witness=proof['witness']
        if witness:
            point=[float(Fraction(x)) for x in witness['parameters']]
            marker='r*' if proof['decision']=='reject' else 'k.'
            axis.plot(point[0],point[1]*1000,marker,markersize=9)
        axis.set(xlim=(1.2,3.),ylim=(center-.06,center+.06),xlabel='Fracture energy (J/m2)',ylabel='Unknown preparation drive (mm)',title=title)
    axes[0].legend(handles=[Patch(color=c,label=l) for l,c in [('Ruled out','#d9d9d9'),('Risk bounded','#68b984'),('Unresolved','#f0ba64')]],fontsize=7)
    new=json.loads((root/'filtered-private-trajectory.json').read_text())['trace']
    old=source_private['trace']
    for trace,label in ((old,'Original finite-grid policy'),(new,'Continuous-region filter')):
        axes[2].plot(range(len(trace)),[r['damage']-trace[1]['damage'] for r in trace],'o-',label=label)
    axes[2].axhline(.02,color='r',linestyle='--',label='Cumulative cap 0.02')
    axes[2].set(xlabel='Observation/action index',ylabel='Actual extra damage over weak reference',title='Single paired regression; no utility claim')
    axes[2].legend(fontsize=7)
    for axis in axes:axis.grid(alpha=.15)
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)


def main():
    project=Path(os.environ['WM_PROJECT_ROOT']);settings=json.loads((project/'config/continuous-risk-probe.json').read_text())
    source=Path(json.loads((project/settings['source_report']).read_text())['run_directory'])
    config=json.loads((source/'config.json').read_text());base=json.loads((source/'physics-config.json').read_text())
    decisions=json.loads((source/'decisions.json').read_text())
    hashes={n:digest(source/n) for n in ('decisions.json','public-episodes.json','private-traces.json','forecasts.json','report.json')}
    root=Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-continuous-risk-probe')
    root.mkdir(parents=True);print(f'OUTPUT={root}',flush=True)
    write_json(root/'config.json',settings);write_json(root/'guard-config.json',config)
    write_json(root/'physics-config.json',base)
    law=InterfaceLaw(**json.loads((project/base['law_config']).read_text())['law']);fixture=Fixture(**base['fixture'])
    sources=[Path(__file__),project/'wmfracture/continuous_risk.py',project/'wmfracture/action_ceiling.py',
        project/'wmfracture/damage_guard.py',project/'wmfracture/compliant_inference.py',
        project/'wmfracture/cohesive.py',project/'wmfracture/compliant_fixture.py',
        project/'wmfracture/measurement_bias.py',project/'scripts/finite_damage_guard.py',
        project/'scripts/history_refinement.py',project/'scripts/replay_continuous_risk_proof.py']
    for path in sources:
        target=root/'source'/path.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in sources})
    shutil.copy2(project/base['law_config'],root/'law-config.json')
    started=time.perf_counter();queries=Queries(root,settings,config,law,fixture)
    initial=audit_recorded(queries,decisions,settings)
    # All retrospective public-query decisions are fixed before private evaluation.
    write_json(root/'recorded-audit-queries.json',queries.rows)
    private=next(p for p in json.loads((source/'private-traces.json').read_text()) if p['episode_id']==settings['regression_episode'])
    write_json(root/'recorded-private-regression.json',private)
    replay=paired_filter_replay(root,queries,settings,config,private,law,fixture,initial['sigma_m'],initial['public_prefix'])
    assert hashes=={n:digest(source/n) for n in hashes}
    write_json(root/'queries.json',queries.rows)
    warm_costs=[{'inputs':verifier.inputs,'bank_cost':bank.model.cost,'warm_action_candidate_queries':bank.action_queries,
        'exact_support_points_evaluated':verifier.point_evaluations} for verifier,bank in queries.cache.values()]
    write_json(root/'warm-costs.json',warm_costs)
    report={'run_directory':str(root),'source_run':str(source),'queries':queries.rows,'filtered_replay':replay,
        'total_seconds':time.perf_counter()-started,'source_files_unchanged':True,'source_sha256':hashes,
        'scope':settings['scope'],'formal_machine_checked_proof':False,'new_method_novelty_validated':False}
    write_json(root/'report.json',report)
    summarize(root,report,private)
    out=project/'artifacts/continuous-risk-probe';out.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','report.json','summary.md','comparison.png','queries.json','warm-costs.json','recorded-audit-queries.json','filtered-public-trajectory.json','filtered-private-trajectory.json','code-sha256.json'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('queries','source_sha256')},indent=2),flush=True)


if __name__=='__main__':
    main()
