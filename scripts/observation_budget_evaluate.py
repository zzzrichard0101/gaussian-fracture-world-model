"""Frozen nested observation budgets with a shared cumulative proof filter."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest, write_json
from continuous_risk_evaluate import new_cases
from continuous_risk_probe import public_inputs, WarmBank
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import PrefixModel, grid, equilibrium_cycle, damage, normalize, weighted_quantile
from wmfracture.measurement_bias import GaussianObservation
from wmfracture.continuous_risk import ContinuousRisk, rational, replay_proof
from wmfracture.observation_budget import BudgetProposal, bounded_noise
from wmfracture.verified_policy import RiskReuseSession, filter_proposal


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class Specimen:
    """Private forward generator, with an action-independent pre-drawn noise stream."""
    def __init__(self,parameters,bias,noise,law,fixture):
        self.energy,self.preparation=parameters;self.bias,self.noise=bias,noise
        self.law,self.fixture=law,fixture
        _,self.maximum=equilibrium_cycle(0.,self.energy,self.preparation,law,fixture)
        self.trace=[]

    def observe(self,action):
        gap,self.maximum=equilibrium_cycle(self.maximum,self.energy,action,self.law,self.fixture)
        error=float(self.noise[len(self.trace)]);observed=float(gap)+self.bias+error
        self.trace.append({'action_m':action,'true_gap_m':float(gap),'maximum_opening_m':float(self.maximum),
            'damage':float(damage(self.maximum,self.energy,self.law)),'noise_m':error,'observed_gap_m':observed})
        return observed


def decide(root,identifier,turn,prefix,sigma,proposer,session,settings,guard,law,fixture):
    """No specimen, private parameter, budget horizon or future label argument."""
    start=time.perf_counter();proposal,record=proposer.plan(prefix,turn)
    data=public_inputs(prefix,sigma,guard,law,fixture,settings['reference_count'])
    bank=WarmBank(data,settings['grid_side'],law,fixture)
    result=filter_proposal(data,proposal,lambda a:bank.points(a,settings['warm_point_count']),
        max_cells=settings['cell_budget'],backtracking_steps=settings['backtracking_steps'],reuse_session=session)
    proof_info=None
    if result['proof'] is not None:
        relative=f'proofs/{identifier}-{turn}.json.gz';path=root/relative
        with gzip.open(path,'wt',encoding='utf-8') as stream:
            json.dump(result['proof'],stream,allow_nan=False)
        proof_info={'file':relative,'sha256':digest(path)}
    row={'trajectory_id':identifier,'round':turn,'public_prefix':prefix,'sigma_m':sigma,
        'proposal_m':proposal,'action_m':result['action_m'],'stopped':result['stopped'],
        'proposal_record':record,'attempts':result['attempts'],'executed_proof':proof_info,
        'search_cache_entries':{name:len(getattr(session.search,name)) for name in
            ('_damage_cache','_point_cache','_observation_cache')},
        'checker_cache_entries':{name:len(getattr(session.checker,name)) for name in
            ('_damage_cache','_point_cache','_observation_cache')},
        'policy_seconds':time.perf_counter()-start}
    row['decision_sha256']=fingerprint(row)
    return row


def collect(root,settings,guard,cases,law,fixture):
    public,private,steps=[],[],[];max_budget=max(settings['budgets'])
    for ci,case in enumerate(cases):
        for ni,condition in enumerate(settings['noise_conditions']):
            for si,sigma in enumerate(settings['gap_noise_sigma_m']):
                for bi,bias in enumerate(settings['actual_bias_m']):
                    context=f'b-{ci:03d}-{ni}-{si}-{bi}'
                    # Same innovations across noise modes; independent streams
                    # across state/sigma/bias. All policies share each stream.
                    seed=settings['seed']+ci*1000+si*10+bi
                    noise=bounded_noise(seed,max_budget+2,sigma,guard['noise_multiple'],condition['latent_rho'])
                    initial=[]
                    for policy in settings['policies']:
                        identifier=context+'-'+policy
                        specimen=Specimen(case['parameters'],bias,noise,law,fixture)
                        proposer=BudgetProposal(policy,settings,guard,law,fixture,sigma);session=RiskReuseSession()
                        actions=[0.,guard['weak_drive_m']];observed=[specimen.observe(a) for a in actions]
                        initial.append(observed.copy());own=[];saved=set();stopped=False
                        def snapshot(budget):
                            public.append({'id':identifier+f'-k{budget}','trajectory_id':identifier,'context_id':context,
                                'policy':policy,'sigma_m':sigma,'noise_condition':condition['name'],'budget':budget,
                                'actions_m':actions.copy(),'observed_gaps_m':observed.copy(),
                                'delivered_observations':len(actions)-2,'budget_fulfilled':len(actions)-2==budget,
                                'stopped':stopped,'decisions_used':len(own),
                                'policy_seconds':sum(d['policy_seconds'] for d in own)})
                            saved.add(budget)
                        for turn in range(max_budget):
                            prefix={'actions_m':actions.copy(),'observed_gaps_m':observed.copy()}
                            decision=decide(root,identifier,turn,prefix,sigma,proposer,session,settings,guard,law,fixture)
                            own.append(decision);steps.append(decision)
                            if decision['stopped']:
                                stopped=True;break
                            actions.append(decision['action_m']);observed.append(specimen.observe(decision['action_m']))
                            if turn+1 in settings['budgets']:snapshot(turn+1)
                        for budget in settings['budgets']:
                            if budget not in saved:snapshot(budget)
                        private.append({'trajectory_id':identifier,'case_id':case['id'],'cluster':case['cluster'],
                            'case_family':case['family'],'parameters':case['parameters'],'actual_bias_m':bias,
                            'noise_condition':condition['name'],'trace':specimen.trace,'noise_seed':seed})
                    assert all(row==initial[0] for row in initial)
        print(f'BUDGET_CASE {ci+1}/{len(cases)} steps={len(steps)}',flush=True)
    write_json(root/'public-episodes.json',public);write_json(root/'private-traces.json',private)
    write_json(root/'decisions.json',steps)


def forecast_stage(root,settings,guard,law,fixture):
    """Public histories only; same Gaussian proxy for every policy/noise mode."""
    public=json.loads((root/'public-episodes.json').read_text());forecasts=[];costs=[]
    for side in settings['forecast_grid_sides']:
        model=PrefixModel(law,fixture,guard['parameter_bounds']);unit=grid(side)
        energy,preparation=model.parameters(unit).T
        for episode in public:
            prediction,maximum=model.predict(unit,episode['actions_m'])
            observation=GaussianObservation('shared',guard['bias_sigma_m'])
            weights=normalize(observation(prediction,episode['observed_gaps_m'],episode['sigma_m']))
            futures=[]
            for action in settings['future_drives_m']:
                gap,after=equilibrium_cycle(maximum,energy,action,law,fixture)
                futures.append({'drive_m':action,'mean_gap_m':float(weights@gap),
                    'q025_gap_m':weighted_quantile(gap,weights,.025),'q975_gap_m':weighted_quantile(gap,weights,.975),
                    'separation_mass':float(weights@(damage(after,energy,law)>=1-1e-12))})
            margin=np.maximum(2*energy/law.peak_traction-np.maximum(preparation,max(episode['actions_m'])),0.)
            forecasts.append({'episode_id':episode['id'],'grid_side':side,'futures':futures,
                'remaining_remote_peak_margin_mean_m':float(weights@margin),'effective_sample_size':float(1/(weights@weights)),
                'bias_moments':observation.bias_moments(prediction,episode['observed_gaps_m'],episode['sigma_m'],weights)})
        costs.append({'grid_side':side,**model.cost,
            'future_action_candidate_queries':len(public)*len(settings['future_drives_m'])*len(unit)})
        print(f'FORECAST side={side} checkpoints={len(public)}',flush=True)
    write_json(root/'forecasts.json',forecasts);write_json(root/'forecast-cost.json',costs)
    names=('public-episodes.json','decisions.json','forecasts.json','private-cases.json','config.json','guard-config.json','code-sha256.json')
    seals={n:digest(root/n) for n in names};write_json(root/'sealed-predictions.json',seals)
    return seals


def evaluate(root,settings,guard,law,fixture,seals):
    # Predictions are fixed before this function first opens private traces.
    public={r['id']:r for r in json.loads((root/'public-episodes.json').read_text())}
    private={r['trajectory_id']:r for r in json.loads((root/'private-traces.json').read_text())}
    steps=json.loads((root/'decisions.json').read_text());by_trajectory={k:[] for k in private}
    replayed=0
    for step in steps:
        assert fingerprint({k:v for k,v in step.items() if k!='decision_sha256'})==step['decision_sha256']
        by_trajectory[step['trajectory_id']].append(step)
        if step['executed_proof']:
            path=root/step['executed_proof']['file'];assert digest(path)==step['executed_proof']['sha256']
            with gzip.open(path,'rt',encoding='utf-8') as stream:proof=json.load(stream)
            data=public_inputs(step['public_prefix'],step['sigma_m'],guard,law,fixture,settings['reference_count'])
            assert replay_proof(proof,expected_inputs=data,expected_action=step['action_m']);replayed+=1
    rows=[]
    for prediction in json.loads((root/'forecasts.json').read_text()):
        episode=public[prediction['episode_id']];truth=private[episode['trajectory_id']]
        trace=truth['trace'][:len(episode['actions_m'])];energy,preparation=truth['parameters']
        assert episode['actions_m']==[r['action_m'] for r in trace]
        assert episode['observed_gaps_m']==[r['observed_gap_m'] for r in trace]
        extra=trace[-1]['damage']-trace[settings['reference_count']-1]['damage']
        actual_bound=abs(truth['actual_bias_m'])<=guard['bias_bound_m'] and all(abs(r['noise_m'])<=guard['noise_multiple']*episode['sigma_m'] for r in trace)
        exact=ContinuousRisk(public_inputs(episode,episode['sigma_m'],guard,law,fixture,settings['reference_count']))
        compatible=exact.point_support(rational(energy),rational(preparation))[0]
        assert actual_bound and compatible and extra<=guard['damage_cap']+1e-12
        mse,coverage,brier,score=[],[],[],[]
        for future in prediction['futures']:
            gap,after=equilibrium_cycle(trace[-1]['maximum_opening_m'],energy,future['drive_m'],law,fixture)
            separated=damage(after,energy,law)>=1-1e-12
            mse.append(float(((future['mean_gap_m']-gap)*1000)**2))
            coverage.append(future['q025_gap_m']-1e-12<=gap<=future['q975_gap_m']+1e-12)
            brier.append(float((future['separation_mass']-separated)**2))
            score.append(float((future['q975_gap_m']-future['q025_gap_m']+
                40*max(future['q025_gap_m']-gap,gap-future['q975_gap_m'],0.))*1000))
        own=by_trajectory[episode['trajectory_id']][:episode['decisions_used']]
        attempts=[a for step in own for a in step['attempts']]
        true_margin=max(2*energy/law.peak_traction-max(preparation,max(episode['actions_m'])),0.)
        rows.append({'episode_id':episode['id'],'trajectory_id':episode['trajectory_id'],'context_id':episode['context_id'],
            'case_id':truth['case_id'],'cluster':truth['cluster'],'case_family':truth['case_family'],
            **{k:episode[k] for k in ('policy','sigma_m','noise_condition','budget','delivered_observations','budget_fulfilled','stopped','policy_seconds')},
            'actual_bias_m':truth['actual_bias_m'],'grid_side':prediction['grid_side'],'extra_damage':extra,
            'cap_violation':bool(extra>guard['damage_cap']+1e-12),'true_parameters_compatible':bool(compatible),
            'actual_errors_within_bounds':bool(actual_bound),'future_mean_squared_error_mm2':float(np.mean(mse)),
            'separation_brier':float(np.mean(brier)),'empirical_95_coverage':float(np.mean(coverage)),
            'interval_score_95_mm':float(np.mean(score)),
            'remaining_peak_margin_squared_error_mm2':float(((prediction['remaining_remote_peak_margin_mean_m']-true_margin)*1000)**2),
            'effective_sample_size':prediction['effective_sample_size'],'bias_std_mm':prediction['bias_moments']['std_m']*1000,
            'verification_queries':len(attempts),'verification_abstentions':sum(a['decision']=='abstain' for a in attempts),
            'verification_rejections':sum(a['decision']=='reject' for a in attempts),
            'probe_actions':sum(a>guard['weak_drive_m'] for a in episode['actions_m'][2:]),
            'search_cache_entries':sum(own[-1]['search_cache_entries'].values()) if own else 0,
            'checker_cache_entries':sum(own[-1]['checker_cache_entries'].values()) if own else 0})
    keys=('noise_condition','sigma_m','policy','budget','grid_side');groups=[]
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        chosen=[r for r in rows if tuple(r[k] for k in keys)==key]
        groups.append({**dict(zip(keys,key)),'checkpoints':len(chosen),'distinct_states':len({r['case_id'] for r in chosen}),
            'gap_rmse_mm':float(np.sqrt(np.mean([r['future_mean_squared_error_mm2'] for r in chosen]))),
            'remaining_peak_margin_rmse_mm':float(np.sqrt(np.mean([r['remaining_peak_margin_squared_error_mm2'] for r in chosen]))),
            'cap_violations':sum(r['cap_violation'] for r in chosen),'maximum_extra_damage':max(r['extra_damage'] for r in chosen),
            'mean_separation_brier':float(np.mean([r['separation_brier'] for r in chosen])),
            'mean_empirical_95_coverage':float(np.mean([r['empirical_95_coverage'] for r in chosen])),
            'mean_interval_score_mm':float(np.mean([r['interval_score_95_mm'] for r in chosen])),
            'mean_policy_ms':float(np.mean([r['policy_seconds'] for r in chosen])*1000),
            'unfilled_budgets':sum(not r['budget_fulfilled'] for r in chosen),
            'mean_delivered_observations':float(np.mean([r['delivered_observations'] for r in chosen])),
            'query_abstentions':sum(r['verification_abstentions'] for r in chosen),
            'mean_probe_actions':float(np.mean([r['probe_actions'] for r in chosen])),
            'maximum_search_cache_entries':max(r['search_cache_entries'] for r in chosen)})
    write_json(root/'evaluation-rows.json',rows)
    assert seals=={n:digest(root/n) for n in seals}
    return groups,rows,replayed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path);args=parser.parse_args()
    project=Path(os.environ['WM_PROJECT_ROOT'])
    settings=json.loads((args.config or project/'config/observation-budget.json').read_text())
    assert settings['budgets']==sorted(set(settings['budgets'])) and min(settings['budgets'])>0
    assert max(settings['budgets'])<=len(settings['fixed_proposals_m']) and settings['reference_count']==2
    assert settings['primary_forecast_side'] in settings['forecast_grid_sides']
    guard=json.loads((project/settings['guard_config']).read_text());guard['grid_side']=settings['grid_side']
    law=InterfaceLaw(**json.loads((project/settings['law_config']).read_text())['law']);fixture=Fixture(**settings['fixture'])
    case_settings=json.loads((project/settings['case_config']).read_text())
    cases=new_cases(case_settings,guard,law,fixture,[])
    root=Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-observation-budget')
    (root/'proofs').mkdir(parents=True);print(f'OUTPUT={root}',flush=True)
    write_json(root/'config.json',settings);write_json(root/'guard-config.json',guard);write_json(root/'private-cases.json',cases)
    write_json(root/'case-config.json',case_settings);shutil.copy2(project/settings['law_config'],root/'law-config.json')
    sources=[Path(__file__),project/'wmfracture/observation_budget.py',project/'wmfracture/continuous_risk.py',
        project/'wmfracture/verified_policy.py',project/'wmfracture/damage_guard.py',project/'wmfracture/action_ceiling.py',
        project/'wmfracture/compliant_inference.py',project/'wmfracture/measurement_bias.py',project/'wmfracture/cohesive.py',
        project/'wmfracture/compliant_fixture.py',project/'scripts/continuous_risk_evaluate.py',
        project/'scripts/continuous_risk_probe.py',project/'scripts/history_refinement.py']
    for p in sources:
        target=root/'source'/p.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in sources})
    old=project/'artifacts/continuous-risk-evaluation/report.json';old_hash=digest(old) if old.exists() else None
    start=time.perf_counter();collect(root,settings,guard,cases,law,fixture);collection=time.perf_counter()-start
    tick=time.perf_counter();seals=forecast_stage(root,settings,guard,law,fixture);forecast_seconds=time.perf_counter()-tick
    groups,rows,replayed=evaluate(root,settings,guard,law,fixture,seals)
    if old_hash:assert digest(old)==old_hash
    report={'run_directory':str(root),'settings':settings,'groups':groups,'distinct_development_states':len(cases),
        'distinct_generation_clusters':len({c['cluster'] for c in cases}),'trajectory_count':len({r['trajectory_id'] for r in rows}),
        'checkpoint_count':len({r['episode_id'] for r in rows}),'forecast_rows':len(rows),
        'collection_seconds':collection,'forecast_seconds':forecast_seconds,'total_seconds':time.perf_counter()-start,
        'executed_allow_proofs_cold_replayed':replayed,'all_error_bounds_and_true_state_compatibility_checked':True,
        'fixed_cumulative_reference_count':2,'nested_budget_prefixes':True,'old_report_sha256':old_hash,
        'old_report_unchanged':True if old_hash else None,'forecast_seals':seals,
        'gaussian_weighting_calibrated_for_correlated_noise':False,'new_method_novelty_validated':False,'scope':settings['scope']}
    write_json(root/'report.json',report);out=project/'artifacts/observation-budget';out.mkdir(parents=True,exist_ok=True)
    for name in ('report.json','evaluation-rows.json','config.json','guard-config.json','case-config.json','sealed-predictions.json','code-sha256.json','forecast-cost.json'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','settings','forecast_seals')},indent=2),flush=True)


if __name__=='__main__':main()
