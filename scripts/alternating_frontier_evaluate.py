"""Stronger clocked baselines on exactly the previous task-pilot histories."""
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest,write_json
from continuous_risk_evaluate import new_cases
from observation_budget_evaluate import Specimen,fingerprint
from response_quadrature_evaluate import forecast_public
from task_aware_pilot import evaluate_risk,asdict_law
from wmfracture.alternating_frontier import AlternatingFrontier
from wmfracture.task_design import TaskController
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.guided_quadrature import GuidedResponseQuadrature
from wmfracture.observation_budget import bounded_noise


def collect(root,settings,guard,cases,law,fixture,reference):
    public,private,decisions=[],[],[];max_budget=max(settings['budgets']);context_index=0;reproduced=0
    gold={(e['context_id'],e['policy'],e['budget']):e for e in reference if e['policy']=='future_task'}
    position_counts={policy:[0]*len(settings['policies']) for policy in settings['policies']}
    for ci,case in enumerate(cases):
        for ni,condition in enumerate(settings['noise_conditions']):
            for si,sigma in enumerate(settings['gap_noise_sigma_m']):
                for bi,bias in enumerate(settings['actual_bias_m']):
                    context=f't-{ci:03d}-{ni}-{si}-{bi}'
                    noise=bounded_noise(settings['seed']+ci*1000+si*10+bi,max_budget+2,sigma,
                        guard['noise_multiple'],condition['latent_rho'])
                    offset=context_index%len(settings['policies']);context_index+=1
                    order=settings['policies'][offset:]+settings['policies'][:offset];initial=[]
                    for position,policy in enumerate(order):
                        position_counts[policy][position]+=1;identifier=context+'-'+policy
                        controller=(TaskController('future_task',law,fixture,guard,settings,sigma) if policy=='future_task' else
                            AlternatingFrontier(policy=='alternating_zero_first',law,fixture,guard,settings,sigma))
                        specimen=Specimen(case['parameters'],bias,noise,law,fixture)
                        actions=[0.,guard['weak_drive_m']];observed=[specimen.observe(a) for a in actions]
                        initial.append(observed.copy());own=[];saved=set();stopped=False
                        def snapshot(budget):
                            nonlocal reproduced
                            episode={'id':identifier+f'-k{budget}','trajectory_id':identifier,'context_id':context,
                                'policy':policy,'sigma_m':sigma,'noise_condition':condition['name'],'budget':budget,
                                'actions_m':actions.copy(),'observed_gaps_m':observed.copy(),
                                'delivered_observations':len(actions)-2,'budget_fulfilled':len(actions)-2==budget,
                                'stopped':stopped,'decisions_used':len(own),'policy_seconds':sum(r['policy_seconds'] for r in own)}
                            if policy=='future_task':
                                prior=gold[(context,policy,budget)]
                                assert episode['actions_m']==prior['actions_m'] and episode['observed_gaps_m']==prior['observed_gaps_m']
                                reproduced+=1
                            public.append(episode);saved.add(budget)
                        for turn in range(max_budget):
                            prefix={'actions_m':actions.copy(),'observed_gaps_m':observed.copy()};tick=time.perf_counter()
                            action,proof,plan=controller.plan(prefix);proof_info=None
                            if proof is not None:
                                relative=f'proofs/{identifier}-{turn}.json.gz';path=root/relative
                                with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(proof,stream,allow_nan=False)
                                proof_info={'file':relative,'sha256':digest(path)}
                            decision={'trajectory_id':identifier,'context_id':context,'policy':policy,'round':turn,
                                'sigma_m':sigma,'public_prefix':prefix,'action_m':action,'plan':plan,
                                'executed_proof':proof_info,'policy_seconds':time.perf_counter()-tick,'policy_position':position}
                            decision['decision_sha256']=fingerprint(decision);decisions.append(decision);own.append(decision)
                            if action is None:stopped=True;break
                            actions.append(action);observed.append(specimen.observe(action))
                            if turn+1 in settings['budgets']:snapshot(turn+1)
                        for budget in settings['budgets']:
                            if budget not in saved:snapshot(budget)
                        private.append({'trajectory_id':identifier,'case_id':case['id'],'cluster':case['cluster'],
                            'case_family':case['family'],'parameters':case['parameters'],'actual_bias_m':bias,
                            'noise_condition':condition['name'],'trace':specimen.trace})
                    assert all(x==initial[0] for x in initial)
        print(f'ALTERNATING_CASE {ci+1}/{len(cases)} decisions={len(decisions)}',flush=True)
    write_json(root/'public-episodes.json',public);write_json(root/'private-traces.json',private);write_json(root/'decisions.json',decisions)
    return {'reproduced_future_task_checkpoints':reproduced,'position_counts':position_counts}


def main():
    project=Path(os.environ['WM_PROJECT_ROOT']);comparison=json.loads((project/'config/alternating-frontier.json').read_text())
    source=json.loads((project/comparison['reference_report']).read_text());settings=dict(source['settings'])
    settings.update(policies=comparison['policies'],scope=comparison['scope'])
    case_settings=json.loads((project/comparison['reference_case_config']).read_text())
    reference=json.loads((project/comparison['reference_episodes']).read_text())
    guard=json.loads((project/settings['guard_config']).read_text());guard['grid_side']=settings['grid_side']
    law=InterfaceLaw(**json.loads((project/settings['law_config']).read_text())['law']);fixture=Fixture(**settings['fixture'])
    cases=new_cases(case_settings,guard,law,fixture,[])
    references={comparison[k]:digest(project/comparison[k]) for k in ('reference_report','reference_case_config','reference_episodes','reference_rows')}
    root=Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-alternating-frontier')
    (root/'proofs').mkdir(parents=True);print(f'OUTPUT={root}',flush=True)
    write_json(root/'config.json',settings);write_json(root/'comparison-config.json',comparison)
    write_json(root/'guard-config.json',guard);write_json(root/'case-config.json',case_settings);write_json(root/'reference-sha256.json',references)
    write_json(root/'private-cases.json',cases)
    paths=[Path(__file__),project/'wmfracture/alternating_frontier.py',project/'wmfracture/task_design.py',
        project/'wmfracture/verified_policy.py',project/'wmfracture/continuous_risk.py',project/'wmfracture/guided_quadrature.py',
        project/'wmfracture/response_quadrature.py',project/'wmfracture/compliant_inference.py',project/'wmfracture/action_ceiling.py',
        project/'wmfracture/damage_guard.py',project/'wmfracture/cohesive.py',project/'wmfracture/compliant_fixture.py',
        project/'wmfracture/observation_budget.py',project/'wmfracture/measurement_bias.py',project/'scripts/task_aware_pilot.py',
        project/'scripts/observation_budget_evaluate.py',project/'scripts/response_quadrature_evaluate.py',
        project/'scripts/continuous_risk_evaluate.py',project/'scripts/history_refinement.py']
    for p in paths:
        target=root/'source'/p.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in paths})
    tick=time.perf_counter();replay=collect(root,settings,guard,cases,law,fixture,reference);collection=time.perf_counter()-tick
    write_json(root/'control-replay.json',replay)
    locked={n:digest(root/n) for n in ('decisions.json','public-episodes.json','private-cases.json','control-replay.json')}
    public=json.loads((root/'public-episodes.json').read_text())
    bundle={'source_run_id':root.name,'episodes':public,'baseline_forecasts':[],
        'source_sha256':locked,'source_settings':settings,'guard':guard,'case_settings':case_settings,'law':asdict_law(law)}
    write_json(root/'inputs.json',bundle)
    tick=time.perf_counter();seals=forecast_public(root,bundle,settings,engine_class=GuidedResponseQuadrature)
    forecast_seconds=time.perf_counter()-tick
    groups,rows,replayed=evaluate_risk(root,bundle,settings,guard,law,fixture,seals)
    # Original evaluation labels are read only after new predictions are sealed.
    old_rows={(r['episode_id'],r['method']):r for r in json.loads((project/comparison['reference_rows']).read_text())}
    reproduced=0
    for row in rows:
        if row['policy']!='future_task':continue
        previous=old_rows[(row['episode_id'],row['method'])]
        for field in ('future_mean_squared_error_mm2','separation_brier'):
            assert np.isclose(row[field],previous[field],atol=1e-14,rtol=1e-12)
        reproduced+=1
    assert references=={name:digest(project/name) for name in references}
    assert locked=={name:digest(root/name) for name in locked}
    report={'run_directory':str(root),'settings':settings,'comparison':comparison,'groups':groups,
        'same_development_states':len(cases),'designed_clusters':len({c['cluster'] for c in cases}),
        'trajectories':len({p['trajectory_id'] for p in public}),'checkpoints':len(public),
        'collection_seconds':collection,'forecast_seconds':forecast_seconds,'executed_allow_proofs_cold_replayed':replayed,
        **replay,'reproduced_future_task_evaluation_rows':reproduced,'original_references_unchanged':True,
        'new_independent_state_test':False,'new_method_novelty_validated':False,'scope':comparison['scope']}
    write_json(root/'report.json',report);out=project/'artifacts/alternating-frontier';out.mkdir(parents=True,exist_ok=True)
    for name in ('report.json','config.json','comparison-config.json','case-config.json','reference-sha256.json','code-sha256.json',
        'control-replay.json','refinement-checks.json','evaluation-rows.json'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('settings','comparison','groups')},indent=2),flush=True)


if __name__=='__main__':main()
