"""Closed-loop stable belief + verified candidates + task-directed observation."""
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
from response_quadrature_evaluate import forecast_public,evaluate as evaluate_predictions
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.continuous_risk import ContinuousRisk,rational,replay_proof
from wmfracture.guided_quadrature import GuidedResponseQuadrature
from wmfracture.observation_budget import bounded_noise
from wmfracture.task_design import TaskController,PublicRiskBank,risk_inputs,PredictiveMixture
from wmfracture.verified_policy import RiskReuseSession,filter_proposal


def baseline_proposal(policy,turn,settings,guard):
    if policy=='rest_repeat':return 0.
    if policy=='weak_repeat':return guard['weak_drive_m']
    if policy=='calibration_alternating':return 0. if turn%2==0 else guard['weak_drive_m']
    if policy=='fixed_ramp':return settings['fixed_proposals_m'][turn]
    raise ValueError('Unknown baseline')


def collect(root,settings,guard,cases,law,fixture):
    public,private,decisions=[],[],[];maximum=max(settings['budgets'])
    for ci,case in enumerate(cases):
        for ni,condition in enumerate(settings['noise_conditions']):
            for si,sigma in enumerate(settings['gap_noise_sigma_m']):
                for bi,bias in enumerate(settings['actual_bias_m']):
                    context=f't-{ci:03d}-{ni}-{si}-{bi}';seed=settings['seed']+ci*1000+si*10+bi
                    noise=bounded_noise(seed,maximum+2,sigma,guard['noise_multiple'],condition['latent_rho']);initials=[]
                    for policy in settings['policies']:
                        identifier=context+'-'+policy
                        specimen=Specimen(case['parameters'],bias,noise,law,fixture)
                        adaptive=policy in ('observation_disagreement','future_task')
                        controller=TaskController(policy,law,fixture,guard,settings,sigma) if adaptive else None
                        session=RiskReuseSession();actions=[0.,guard['weak_drive_m']]
                        observed=[specimen.observe(a) for a in actions];initials.append(observed.copy())
                        own=[];saved=set();stopped=False
                        def snapshot(budget):
                            public.append({'id':identifier+f'-k{budget}','trajectory_id':identifier,'context_id':context,
                                'policy':policy,'sigma_m':sigma,'noise_condition':condition['name'],'budget':budget,
                                'actions_m':actions.copy(),'observed_gaps_m':observed.copy(),
                                'delivered_observations':len(actions)-2,'budget_fulfilled':len(actions)-2==budget,
                                'stopped':stopped,'decisions_used':len(own),'policy_seconds':sum(r['policy_seconds'] for r in own)})
                            saved.add(budget)
                        for turn in range(maximum):
                            prefix={'actions_m':actions.copy(),'observed_gaps_m':observed.copy()};tick=time.perf_counter()
                            if adaptive:action,proof,record=controller.plan(prefix)
                            else:
                                proposal=baseline_proposal(policy,turn,settings,guard)
                                inputs=risk_inputs(prefix,sigma,guard,law,fixture);bank=PublicRiskBank(inputs,guard,law,fixture)
                                filtered=filter_proposal(inputs,proposal,bank.warm,max_cells=settings['cell_budget'],
                                    backtracking_steps=settings['backtracking_steps'],reuse_session=session)
                                action,proof=filtered['action_m'],filtered['proof']
                                record={'proposal_m':proposal,'attempts':filtered['attempts'],'acquisition_abstained':False}
                            proof_info=None
                            if proof:
                                relative=f'proofs/{identifier}-{turn}.json.gz';path=root/relative
                                with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(proof,stream,allow_nan=False)
                                proof_info={'file':relative,'sha256':digest(path)}
                            decision={'trajectory_id':identifier,'context_id':context,'policy':policy,'round':turn,
                                'sigma_m':sigma,'public_prefix':prefix,'action_m':action,'plan':record,
                                'executed_proof':proof_info,'policy_seconds':time.perf_counter()-tick}
                            decision['decision_sha256']=fingerprint(decision);decisions.append(decision);own.append(decision)
                            if action is None:stopped=True;break
                            actions.append(action);observed.append(specimen.observe(action))
                            if turn+1 in settings['budgets']:snapshot(turn+1)
                        for budget in settings['budgets']:
                            if budget not in saved:snapshot(budget)
                        private.append({'trajectory_id':identifier,'case_id':case['id'],'cluster':case['cluster'],
                            'case_family':case['family'],'parameters':case['parameters'],'actual_bias_m':bias,
                            'noise_condition':condition['name'],'trace':specimen.trace})
                    assert all(x==initials[0] for x in initials)
        print(f'TASK_CASE {ci+1}/{len(cases)} decisions={len(decisions)}',flush=True)
    write_json(root/'public-episodes.json',public);write_json(root/'private-traces.json',private)
    write_json(root/'decisions.json',decisions)


def score_audit(root,settings,guard,law,fixture):
    """Refine chosen public beliefs, without changing any executed decision."""
    decisions=json.loads((root/'decisions.json').read_text());audit=[]
    for decision in decisions:
        if decision['policy'] not in ('observation_disagreement','future_task'):continue
        parts=decision['context_id'].split('-');plan=decision['plan']
        if (int(parts[1]) not in (0,4,8) or int(parts[4])!=1
                or decision['round'] not in (0,7) or not plan['scores'] or plan['acquisition_abstained']):continue
        prefix=decision['public_prefix']
        model=GuidedResponseQuadrature(law,fixture,guard['parameter_bounds'],prefix['actions_m'],prefix['observed_gaps_m'],
            decision['sigma_m'],guard['bias_sigma_m'])
        reference=model.integrate(settings['future_drives_m'],**settings['belief_levels'][-1])
        mixture=PredictiveMixture(model,reference,settings);values=[];stable=True
        for item in plan['scores']:
            if decision['policy']=='future_task':value,detail=mixture.task_score(item['action_m'],settings);stable=stable and detail['stable']
            else:value=mixture.disagreement(item['action_m'])
            values.append({'action_m':item['action_m'],'score':value})
        if stable:
            maximum=max(r['score'] for r in values)
            minimum=settings['minimum_gain_mm2'] if decision['policy']=='future_task' else 1e-8
            tolerance=max(plan.get('empirical_rank_tolerance_mm2',1e-12),minimum)
            selected=next(r['score'] for r in values if r['action_m']==decision['action_m'])
            regret=max(0.,maximum-selected)
            change=max(abs(a['score']-b['score']) for a,b in zip(values,plan['scores']))
        else:regret=None;change=None
        audit.append({'trajectory_id':decision['trajectory_id'],'round':decision['round'],'policy':decision['policy'],
            'reference_scores_stable':stable,'maximum_score_change':change,'reference_score_regret':regret,
            'selected_action_still_best_or_near_tied':bool(stable and regret<=tolerance),
            'action_m':decision['action_m'],'reference_scores':values})
    write_json(root/'score-audits.json',audit);return audit


def evaluate_risk(root,bundle,settings,guard,law,fixture,seals):
    groups,rows=evaluate_predictions(root,bundle,settings,seals)
    public={e['id']:e for e in bundle['episodes']}
    private={e['trajectory_id']:e for e in json.loads((root/'private-traces.json').read_text())}
    decisions=json.loads((root/'decisions.json').read_text());own={k:[] for k in private};replayed=0
    for decision in decisions:
        assert fingerprint({k:v for k,v in decision.items() if k!='decision_sha256'})==decision['decision_sha256']
        own[decision['trajectory_id']].append(decision)
        if decision['executed_proof']:
            path=root/decision['executed_proof']['file'];assert digest(path)==decision['executed_proof']['sha256']
            with gzip.open(path,'rt',encoding='utf-8') as stream:proof=json.load(stream)
            assert replay_proof(proof,expected_inputs=risk_inputs(decision['public_prefix'],decision['sigma_m'],guard,law,fixture),
                expected_action=decision['action_m']);replayed+=1
    for row in rows:
        e=public[row['episode_id']];p=private[e['trajectory_id']];trace=p['trace'][:len(e['actions_m'])]
        assert e['actions_m']==[r['action_m'] for r in trace]
        assert e['observed_gaps_m']==[r['observed_gap_m'] for r in trace]
        extra=trace[-1]['damage']-trace[1]['damage'];energy,history=p['parameters']
        within=abs(p['actual_bias_m'])<=guard['bias_bound_m'] and all(abs(r['noise_m'])<=guard['noise_multiple']*e['sigma_m'] for r in trace)
        compatible=ContinuousRisk(risk_inputs(e,e['sigma_m'],guard,law,fixture)).point_support(rational(energy),rational(history))[0]
        assert within and compatible and extra<=guard['damage_cap']+1e-12
        used=own[e['trajectory_id']][:e['decisions_used']];attempts=[]
        for d in used:
            plan=d['plan'];attempts+=plan.get('attempts',[])+plan.get('frontier_attempts',[])
            for bank in plan.get('candidate_checks',[]):attempts+=bank['attempts']
        row.update({'actual_bias_m':p['actual_bias_m'],'extra_damage':extra,'cap_violation':bool(extra>guard['damage_cap']+1e-12),
            'actual_errors_within_bounds':bool(within),'true_parameters_compatible':bool(compatible),
            **{k:e[k] for k in ('delivered_observations','budget_fulfilled','stopped','policy_seconds')},
            'acquisition_abstentions':sum(d['plan'].get('acquisition_abstained',False) for d in used),
            'risk_query_abstentions':sum(a['decision']=='abstain' for a in attempts),'risk_queries':len(attempts),
            'belief_seconds':sum(d['plan'].get('belief_seconds',0.) for d in used),
            'score_seconds':sum(d['plan'].get('score_seconds',0.) for d in used),
            'zero_actions':sum(a==0. for a in e['actions_m'][2:]),
            'probe_actions':sum(a>guard['weak_drive_m'] for a in e['actions_m'][2:])})
    fields=('noise_condition','sigma_m','policy','budget','method');groups=[]
    for key in sorted({tuple(r[k] for k in fields) for r in rows}):
        selected=[r for r in rows if tuple(r[k] for k in fields)==key]
        groups.append({**dict(zip(fields,key)),'checkpoints':len(selected),
            'gap_rmse_mm':float(np.sqrt(np.mean([r['future_mean_squared_error_mm2'] for r in selected]))),
            'mean_separation_brier':float(np.mean([r['separation_brier'] for r in selected])),
            'cap_violations':sum(r['cap_violation'] for r in selected),'maximum_extra_damage':max(r['extra_damage'] for r in selected),
            'unfilled_budgets':sum(not r['budget_fulfilled'] for r in selected),
            'acquisition_abstentions':sum(r['acquisition_abstentions'] for r in selected),
            'empirical_refinement_failures':sum(not r['empirical_refinement_pass'] for r in selected),
            'mean_policy_ms':float(np.mean([r['policy_seconds'] for r in selected])*1000),
            'mean_zero_actions':float(np.mean([r['zero_actions'] for r in selected])),
            'mean_probe_actions':float(np.mean([r['probe_actions'] for r in selected]))})
    write_json(root/'evaluation-rows.json',rows)
    return groups,rows,replayed


def main():
    project=Path(os.environ['WM_PROJECT_ROOT']);extra=json.loads((project/'config/task-aware-pilot.json').read_text())
    settings=json.loads((project/extra['base_config']).read_text());settings.update(extra)
    settings.update(levels=[{'name':'guided_32',**settings['belief_levels'][1]},{'name':'guided_64',**settings['belief_levels'][2]}],
        primary_level='guided_64',event_mass_change_tolerance=settings['event_change_tolerance'],adaptive_audit_case_indices=[],
        adaptive_audit_budget=8,adaptive_audit_actual_bias_m=0.)
    guard=json.loads((project/settings['guard_config']).read_text());guard['grid_side']=settings['grid_side']
    law=InterfaceLaw(**json.loads((project/settings['law_config']).read_text())['law']);fixture=Fixture(**settings['fixture'])
    old_cases_settings=json.loads((project/settings['case_config']).read_text())
    old_cases=new_cases(old_cases_settings,guard,law,fixture,[])
    case_settings=dict(old_cases_settings);case_settings.update(seed=settings['case_seed'],families=settings['case_families'])
    cases=new_cases(case_settings,guard,law,fixture,old_cases)
    root=Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-task-aware-pilot')
    (root/'proofs').mkdir(parents=True);print(f'OUTPUT={root}',flush=True)
    write_json(root/'config.json',settings);write_json(root/'guard-config.json',guard);write_json(root/'case-config.json',case_settings)
    write_json(root/'private-cases.json',cases)
    files=[Path(__file__),project/'wmfracture/task_design.py',project/'wmfracture/guided_quadrature.py',
        project/'wmfracture/response_quadrature.py',project/'wmfracture/verified_policy.py',project/'wmfracture/continuous_risk.py',
        project/'wmfracture/compliant_inference.py',project/'wmfracture/action_ceiling.py',project/'wmfracture/damage_guard.py',
        project/'wmfracture/measurement_bias.py',project/'wmfracture/cohesive.py',project/'wmfracture/compliant_fixture.py',
        project/'wmfracture/observation_budget.py',project/'scripts/observation_budget_evaluate.py',
        project/'scripts/response_quadrature_evaluate.py',project/'scripts/continuous_risk_evaluate.py',project/'scripts/history_refinement.py']
    for p in files:
        target=root/'source'/p.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in files})
    old_reports={n:digest(project/'artifacts'/n/'report.json') for n in ('observation-budget','guided-quadrature')}
    tick=time.perf_counter();collect(root,settings,guard,cases,law,fixture);collection=time.perf_counter()-tick
    tick=time.perf_counter();audits=score_audit(root,settings,guard,law,fixture);audit_seconds=time.perf_counter()-tick
    public=json.loads((root/'public-episodes.json').read_text())
    locked={n:digest(root/n) for n in ('decisions.json','public-episodes.json','score-audits.json','private-cases.json')}
    bundle={'source_run_id':root.name,'episodes':public,'baseline_forecasts':[],
        'source_sha256':locked,'source_settings':settings,'guard':guard,'case_settings':case_settings,'law':asdict_law(law)}
    write_json(root/'inputs.json',bundle)
    tick=time.perf_counter();seals=forecast_public(root,bundle,settings,engine_class=GuidedResponseQuadrature)
    forecast_seconds=time.perf_counter()-tick
    groups,rows,replayed=evaluate_risk(root,bundle,settings,guard,law,fixture,seals)
    assert locked=={n:digest(root/n) for n in locked}
    assert old_reports=={n:digest(project/'artifacts'/n/'report.json') for n in old_reports}
    report={'run_directory':str(root),'settings':settings,'groups':groups,'distinct_new_states':len(cases),
        'designed_clusters':len({c['cluster'] for c in cases}),'trajectories':len({r['trajectory_id'] for r in public}),
        'checkpoints':len(public),'collection_seconds':collection,'score_audit_seconds':audit_seconds,
        'forecast_seconds':forecast_seconds,'executed_allow_proofs_cold_replayed':replayed,
        'score_audits':len(audits),'score_audit_ranking_changes':sum(not r['selected_action_still_best_or_near_tied'] for r in audits),
        'new_states_disjoint_from_previous_24':True,'old_results_unchanged':True,
        'new_method_novelty_validated':False,'global_action_optimality_claimed':False,'scope':settings['scope']}
    write_json(root/'report.json',report);out=project/'artifacts/task-aware-pilot';out.mkdir(parents=True,exist_ok=True)
    for name in ('report.json','config.json','guard-config.json','case-config.json','code-sha256.json','evaluation-rows.json',
        'score-audits.json','refinement-checks.json','sealed-predictions.json'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','settings')},indent=2),flush=True)


def asdict_law(law):
    from dataclasses import asdict
    return asdict(law)


if __name__=='__main__':main()
