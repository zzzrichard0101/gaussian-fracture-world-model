"""Frozen new-state evaluation of standard finite and verified action filters."""
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest,write_json
from continuous_risk_probe import public_inputs,WarmBank
from finite_damage_guard import SyntheticSpecimen,forecast
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture,nominal_remote_peak
from wmfracture.compliant_inference import equilibrium_cycle,damage
from wmfracture.joint_state import matched_history
from wmfracture.damage_guard import FiniteGuard
from wmfracture.continuous_risk import ContinuousRisk,rational,replay_proof
from wmfracture.verified_policy import filter_proposal


class EvaluationSpecimen(SyntheticSpecimen):
    def __init__(self,*args,noise_family,noise_multiple,**kwargs):
        super().__init__(*args,**kwargs)
        self.noise_family,self.noise_multiple=noise_family,noise_multiple

    def observe(self,action):
        gap,self.maximum=equilibrium_cycle(self.maximum,self.energy,action,self.law,self.fixture)
        noise=float(self.rng.normal(0,self.sigma))
        if self.noise_family=='bounded_normal':
            while abs(noise)>self.noise_multiple*self.sigma:
                noise=float(self.rng.normal(0,self.sigma))
        elif self.noise_family!='gaussian':
            raise ValueError('Unknown predeclared noise family')
        observed=float(gap)+self.bias+noise
        self.trace.append({'action_m':action,'true_gap_m':float(gap),'maximum_opening_m':float(self.maximum),
            'damage':float(damage(self.maximum,self.energy,self.law)),'noise_m':noise,'observed_gap_m':observed})
        return observed


def new_cases(settings,config,law,fixture,old):
    rng=np.random.default_rng(settings['seed']);bounds=np.asarray(config['parameter_bounds']);cases=[]
    def add(parameters,family,cluster):
        cases.append({'id':f'new-{len(cases):03d}','parameters':list(map(float,parameters)),
            'family':family,'cluster':cluster,'in_prior':True})
    for i in range(settings['families']['uniform']):
        unit=rng.uniform(.02,.98,2)
        add(bounds[:,0]+unit*np.ptp(bounds,axis=1),'uniform',f'uniform-{i}')
    for i in range(settings['families']['near_initiation']):
        add([rng.uniform(1.23,2.97),rng.uniform(*settings['near_initiation_drive_m'])],
            'near_initiation',f'near-{i}')
    known={'law':{'area':law.area,'tensile_stiffness':law.tensile_stiffness,
        'compressive_stiffness':law.compressive_stiffness,'peak_traction':law.peak_traction}}
    for i in range(settings['families']['matched_pairs']):
        ratio=rng.uniform(*settings['matched_ratios'])
        for energy in (rng.uniform(1.27,1.72),rng.uniform(2.2,2.85)):
            material=replace(law,fracture_energy=float(energy))
            history=matched_history(known,float(energy),ratio)
            add([energy,nominal_remote_peak(material,fixture,history)],'matched_pair',f'pair-{i}')
    old_parameters={tuple(c['parameters']) for c in old}
    assert not any(tuple(c['parameters']) in old_parameters for c in cases)
    assert len({tuple(c['parameters']) for c in cases})==len(cases)
    for case in cases:
        unit=(np.asarray(case['parameters'])-bounds[:,0])/np.ptp(bounds,axis=1)
        assert (unit>0).all() and (unit<1).all()
        assert not np.allclose(unit*(config['grid_side']-1),np.round(unit*(config['grid_side']-1)),atol=1e-10,rtol=0)
    return cases


def decide(root,identifier,turn,policy,controller,prefix,sigma,settings,config,law,fixture):
    started=time.perf_counter();proof_info=None;filter_info=None;proposal_record=None
    if policy=='rest_only':
        action=0.;proposal=action
    elif policy=='fixed_probe':
        action=config['fixed_probes_m'][turn];proposal=action
    else:
        proposal,proposal_record=controller.plan(prefix['actions_m'],prefix['observed_gaps_m'],
            'continuous_guard' if policy=='verified_filter' else policy,turn)
        action=proposal
        if policy=='verified_filter':
            inputs=public_inputs(prefix,sigma,config,law,fixture,2)
            warm_started=time.perf_counter();bank=WarmBank(inputs,settings['grid_side'],law,fixture)
            bank_seconds=time.perf_counter()-warm_started
            filtered=filter_proposal(inputs,proposal,lambda a:bank.points(a,settings['warm_point_count']),
                max_cells=settings['cell_budget'],backtracking_steps=settings['backtracking_steps'])
            action=filtered['action_m']
            filter_info={'stopped':filtered['stopped'],'attempts':filtered['attempts'],
                'warm_bank_seconds':bank_seconds,'warm_bank_cost':dict(bank.model.cost),
                'warm_action_candidate_queries':bank.action_queries}
            if action is not None:
                # Every actually executed filter action retains a self-contained
                # cover. Intermediate covers were replayed but are not archived.
                relative=f'executed-proofs/{identifier}-round-{turn}.json'
                path=root/relative;path.parent.mkdir(exist_ok=True)
                write_json(path,filtered['proof'])
                proof_info={'file':relative,'sha256':digest(path),'decision':'allow',
                    'upper_damage':filtered['proof']['upper_damage']}
    return {'episode_id':identifier,'round':turn,'policy':policy,'public_prefix':prefix,
        'sigma_m':sigma,'proposal_m':proposal,'action_m':action,'proposal_record':proposal_record,
        'filter':filter_info,'executed_proof':proof_info,'policy_seconds':time.perf_counter()-started}


def collect(root,settings,config,cases,law,fixture):
    public,private,steps=[],[],[]
    for ci,case in enumerate(cases):
        for fi,family in enumerate(settings['noise_families']):
            for si,sigma in enumerate(settings['gap_noise_sigma_m']):
                for bi,bias in enumerate(settings['actual_bias_m']):
                    context=f'eval-{ci:03d}-{fi}-{si}-{bi}'
                    seed=settings['seed']+100000+ci*1000+fi*100
                    initials=[]
                    for policy in settings['policies']:
                        identifier=context+'-'+policy
                        specimen=EvaluationSpecimen(case['parameters'],bias,sigma,seed,law,fixture,
                            noise_family=family,noise_multiple=config['noise_multiple'])
                        controller=None if policy in ('rest_only','fixed_probe') else FiniteGuard(law,fixture,config,sigma)
                        actions=[0.,config['weak_drive_m']];observed=[specimen.observe(a) for a in actions]
                        initials.append(observed.copy());stopped=False
                        for turn in range(config['decision_rounds']):
                            prefix={'actions_m':actions.copy(),'observed_gaps_m':observed.copy()}
                            decision=decide(root,identifier,turn,policy,controller,prefix,sigma,settings,config,law,fixture)
                            decision['decision_sha256']=hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()
                            steps.append(decision)
                            if decision['action_m'] is None:
                                stopped=True;break
                            actions.append(decision['action_m']);observed.append(specimen.observe(decision['action_m']))
                        public.append({'id':identifier,'context_id':context,'policy':policy,'sigma_m':sigma,
                            'actions_m':actions,'observed_gaps_m':observed})
                        private.append({'episode_id':identifier,'case_id':case['id'],'case_family':case['family'],
                            'cluster':case['cluster'],'parameters':case['parameters'],'actual_bias_m':bias,
                            'noise_family':family,'trace':specimen.trace,'stopped':stopped})
                    assert all(row==initials[0] for row in initials)
        print(f'EVALUATION_CASE {ci+1}/{len(cases)}',flush=True)
    indexed={(e['context_id'],e['policy']):e for e in public}
    for context in {e['context_id'] for e in public}:
        a,b=indexed[(context,'full_guard')],indexed[(context,'witness_guard')]
        assert a['actions_m']==b['actions_m'] and a['observed_gaps_m']==b['observed_gaps_m']
    write_json(root/'public-episodes.json',public);write_json(root/'private-traces.json',private)
    write_json(root/'decisions.json',steps)


def evaluate(root,settings,config,law,fixture,seals):
    public={e['id']:e for e in json.loads((root/'public-episodes.json').read_text())}
    private={p['episode_id']:p for p in json.loads((root/'private-traces.json').read_text())}
    steps=json.loads((root/'decisions.json').read_text());predictions=json.loads((root/'forecasts.json').read_text())
    by_episode={identifier:[] for identifier in public}
    for step in steps:
        body={k:v for k,v in step.items() if k!='decision_sha256'}
        assert step['decision_sha256']==hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
        if step['executed_proof']:
            proof=step['executed_proof'];assert digest(root/proof['file'])==proof['sha256']
            result=json.loads((root/proof['file']).read_text())
            replay_proof(result,expected_inputs=public_inputs(step['public_prefix'],step['sigma_m'],config,law,fixture,2),
                expected_action=step['action_m'])
        by_episode[step['episode_id']].append(step)
    rows=[]
    for prediction in predictions:
        e,p=public[prediction['episode_id']],private[prediction['episode_id']]
        own_steps=by_episode[e['id']];trace=p['trace'];energy,preparation=p['parameters']
        extra=trace[-1]['damage']-trace[1]['damage'];violated=extra>config['damage_cap']+1e-12
        radius=config['noise_multiple']*e['sigma_m']
        actual_bounds=abs(p['actual_bias_m'])<=config['bias_bound_m'] and all(abs(t['noise_m'])<=radius for t in trace)
        exact=ContinuousRisk(public_inputs(e,e['sigma_m'],config,law,fixture,2))
        compatible=exact.point_support(rational(energy),rational(preparation))[0]
        if e['policy']=='verified_filter' and actual_bounds and compatible and violated:
            raise AssertionError('Conditional verifier failed on an in-assumption new state')
        errors,coverage,scores,briers=[],[],[],[]
        for future in prediction['futures']:
            truth,after=equilibrium_cycle(trace[-1]['maximum_opening_m'],energy,future['drive_m'],law,fixture)
            separated=bool(damage(after,energy,law)>=1-1e-12)
            errors.append(float(((future['mean_gap_m']-truth)*1000)**2))
            coverage.append(future['q025_gap_m']-1e-12<=truth<=future['q975_gap_m']+1e-12)
            scores.append(float((future['q975_gap_m']-future['q025_gap_m']+
                40*max(future['q025_gap_m']-truth,truth-future['q975_gap_m'],0.))*1000))
            briers.append((future['separation_mass']-separated)**2)
        attempts=[a for step in own_steps if step['filter'] for a in step['filter']['attempts']]
        policy_cycles=0;verifier_cells=sum(a['cells'] for a in attempts)
        for step in own_steps:
            record=step['proposal_record']
            if record:
                cost=record['cost'];policy_cycles+=cost['information_action_candidate_queries']+cost['guard_action_candidate_queries']+cost['witness_construction_queries']
        if own_steps[-1]['proposal_record']:
            policy_cycles+=own_steps[-1]['proposal_record']['cost']['equilibrium_cycle_evaluations']
        warm_cycles=sum(step['filter']['warm_bank_cost']['equilibrium_cycle_evaluations']+
            step['filter']['warm_action_candidate_queries'] for step in own_steps if step['filter'])
        rows.append({'episode_id':e['id'],'case_id':p['case_id'],'case_family':p['case_family'],'cluster':p['cluster'],
            'noise_family':p['noise_family'],'sigma_m':e['sigma_m'],'actual_bias_m':p['actual_bias_m'],'policy':e['policy'],
            'extra_damage':extra,'cap_violation':bool(violated),'actual_errors_within_bounds':bool(actual_bounds),
            'true_parameters_observation_compatible':bool(compatible),'stopped_without_execution':p['stopped'],
            'completed_decisions':len(e['actions_m'])-2,'probe_actions':sum(a>config['weak_drive_m'] for a in e['actions_m'][2:]),
            'rest_actions':sum(a==0. for a in e['actions_m'][2:]),'observations_used':len(e['actions_m']),
            'future_mean_squared_error_mm2':float(np.mean(errors)),'empirical_95_coverage':float(np.mean(coverage)),
            'interval_score_95_mm':float(np.mean(scores)),'separation_brier':float(np.mean(briers)),
            'policy_seconds':sum(s['policy_seconds'] for s in own_steps),'logical_cycle_queries':policy_cycles+warm_cycles,
            'verification_queries':len(attempts),'verification_abstentions':sum(a['decision']=='abstain' for a in attempts),
            'verification_rejections':sum(a['decision']=='reject' for a in attempts),'verifier_cells':verifier_cells,
            'proof_replay_seconds':sum(a['proof_replay_seconds'] for a in attempts),
            'empty_finite_set_steps':sum(bool(s['proposal_record'] and s['proposal_record']['empty_set']) for s in own_steps)})
    groups=[]
    for key in sorted({(r['noise_family'],r['sigma_m'],r['actual_bias_m'],r['policy']) for r in rows}):
        chosen=[r for r in rows if (r['noise_family'],r['sigma_m'],r['actual_bias_m'],r['policy'])==key]
        groups.append(dict(zip(('noise_family','sigma_m','actual_bias_m','policy'),key),episodes=len(chosen),
            cap_violations=sum(r['cap_violation'] for r in chosen),maximum_extra_damage=max(r['extra_damage'] for r in chosen),
            assumption_bound_violations=sum(not r['actual_errors_within_bounds'] for r in chosen),
            gap_rmse_mm=float(np.sqrt(np.mean([r['future_mean_squared_error_mm2'] for r in chosen]))),
            mean_interval_score_mm=float(np.mean([r['interval_score_95_mm'] for r in chosen])),
            mean_brier=float(np.mean([r['separation_brier'] for r in chosen])),
            mean_policy_seconds=float(np.mean([r['policy_seconds'] for r in chosen])),
            mean_logical_cycle_queries=float(np.mean([r['logical_cycle_queries'] for r in chosen])),
            mean_verifier_cells=float(np.mean([r['verifier_cells'] for r in chosen])),
            stopped_episodes=sum(r['stopped_without_execution'] for r in chosen),
            query_abstentions=sum(r['verification_abstentions'] for r in chosen),
            mean_probe_actions=float(np.mean([r['probe_actions'] for r in chosen]))))
    write_json(root/'evaluation-rows.json',rows)
    assert seals=={n:digest(root/n) for n in seals}
    return groups,rows


def main():
    project=Path(os.environ['WM_PROJECT_ROOT']);settings=json.loads((project/'config/continuous-risk-evaluation.json').read_text())
    config=json.loads((project/settings['guard_config']).read_text());config['grid_side']=settings['grid_side']
    config['policies']=settings['policies'];config['future_drives_m']=settings['future_drives_m']
    old_root=Path(json.loads((project/settings['old_case_report']).read_text())['run_directory'])
    old=json.loads((old_root/'private-cases.json').read_text())
    old_hashes={n:digest(old_root/n) for n in ('private-cases.json','report.json','forecasts.json')}
    root=Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-continuous-risk-evaluation')
    root.mkdir(parents=True);print(f'OUTPUT={root}',flush=True)
    write_json(root/'config.json',settings);write_json(root/'guard-config.json',config)
    sources=[Path(__file__),project/'wmfracture/verified_policy.py',project/'wmfracture/continuous_risk.py',
        project/'wmfracture/damage_guard.py',project/'wmfracture/action_ceiling.py',project/'wmfracture/compliant_inference.py',
        project/'wmfracture/measurement_bias.py',project/'wmfracture/cohesive.py',project/'wmfracture/compliant_fixture.py',
        project/'scripts/finite_damage_guard.py',project/'scripts/continuous_risk_probe.py',project/'scripts/history_refinement.py']
    for path in sources:
        target=root/'source'/path.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in sources})
    law=InterfaceLaw(**json.loads((project/settings['law_config']).read_text())['law']);fixture=Fixture(**settings['fixture'])
    shutil.copy2(project/settings['law_config'],root/'law-config.json')
    cases=new_cases(settings,config,law,fixture,old);write_json(root/'private-cases.json',cases)
    started=time.perf_counter();collect(root,settings,config,cases,law,fixture);collection=time.perf_counter()-started
    tick=time.perf_counter();seals=forecast(root,config,config,law,fixture);forecast_seconds=time.perf_counter()-tick
    groups,rows=evaluate(root,settings,config,law,fixture,seals)
    assert old_hashes=={n:digest(old_root/n) for n in old_hashes}
    report={'run_directory':str(root),'settings':settings,'groups':groups,'distinct_new_states':len(cases),
        'episodes':len(rows),'collection_seconds':collection,'forecast_seconds':forecast_seconds,
        'total_seconds':time.perf_counter()-started,'old_results_unchanged':True,'old_sha256':old_hashes,
        'new_state_parameters_disjoint_from_previous_26':True,'all_executed_allow_proofs_replayed':True,
        'intermediate_covers_replayed_but_not_archived':True,'full_and_witness_trajectories_identical':True,
        'scope':settings['scope'],'new_method_novelty_validated':False,'seals':seals}
    write_json(root/'report.json',report)
    out=project/'artifacts/continuous-risk-evaluation';out.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','guard-config.json','report.json','evaluation-rows.json','sealed-predictions.json','forecast-cost.json','code-sha256.json'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','settings','seals','old_sha256')},indent=2),flush=True)


if __name__=='__main__':main()
