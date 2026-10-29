"""Frozen public-history integration replay, then separate synthetic evaluation."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest,write_json
from continuous_risk_evaluate import new_cases
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import equilibrium_cycle,damage
from wmfracture.response_quadrature import ResponseQuadrature


def bundle_from_source(root):
    names=('public-episodes.json','forecasts.json','config.json','guard-config.json','case-config.json','law-config.json')
    return {'source_run_id':root.name,'source_sha256':{n:digest(root/n) for n in names},
        'episodes':json.loads((root/'public-episodes.json').read_text()),
        'baseline_forecasts':json.loads((root/'forecasts.json').read_text()),
        'source_settings':json.loads((root/'config.json').read_text()),
        'guard':json.loads((root/'guard-config.json').read_text()),
        'case_settings':json.loads((root/'case-config.json').read_text()),
        'law':json.loads((root/'law-config.json').read_text())['law']}


def audit_selected(episode,settings,bundle):
    parts=episode['context_id'].split('-')
    bias_index=bundle['source_settings']['actual_bias_m'].index(settings['adaptive_audit_actual_bias_m'])
    return (episode['budget']==settings['adaptive_audit_budget'] and int(parts[1]) in settings['adaptive_audit_case_indices']
        and int(parts[4])==bias_index)


def forecast_public(root,bundle,settings,engine_class=ResponseQuadrature):
    law=InterfaceLaw(**bundle['law']);fixture=Fixture(**bundle['source_settings']['fixture']);guard=bundle['guard']
    future=bundle['source_settings']['future_drives_m'];rows=[];comparisons=[];audits=[]
    for index,episode in enumerate(bundle['episodes']):
        model=engine_class(law,fixture,guard['parameter_bounds'],episode['actions_m'],episode['observed_gaps_m'],
            episode['sigma_m'],guard['bias_sigma_m'])
        own=[];last=None
        for level in settings['levels']:
            tick=time.perf_counter()
            result=model.integrate(future,**{k:v for k,v in level.items() if k!='name'})
            prediction=model.forecast(result,future)
            record={'episode_id':episode['id'],'method':level['name'],'futures':prediction,
                'log_integral':result.log_integral,'seconds':time.perf_counter()-tick,'diagnostics':result.diagnostics}
            rows.append(record);own.append(record);last=result
        for a,b in zip(own[:-1],own[1:]):
            means=[abs(x['mean_gap_m']-y['mean_gap_m']) for x,y in zip(a['futures'],b['futures'])]
            events=[abs(x['separation_mass']-y['separation_mass']) for x,y in zip(a['futures'],b['futures'])]
            log_change=abs(a['log_integral']-b['log_integral'])
            comparisons.append({'episode_id':episode['id'],'from':a['method'],'to':b['method'],
                'maximum_mean_change_m':max(means),'maximum_event_mass_change':max(events),'log_integral_change':log_change,
                'empirical_tolerance_pass':bool(max(means)<=settings['mean_change_tolerance_m'] and
                    max(events)<=settings['event_mass_change_tolerance'] and log_change<=settings['log_integral_change_tolerance'])})
        if audit_selected(episode,settings,bundle):
            tick=time.perf_counter();reference=model.adaptive_reference(future,last.log_integral,**settings['adaptive_reference'])
            record={'episode_id':episode['id'],'method':'adaptive_outer','futures':reference['futures'],
                'log_integral':reference['log_integral'],'seconds':time.perf_counter()-tick,
                'diagnostics':{k:v for k,v in reference.items() if k not in ('futures','log_integral')}}
            rows.append(record)
            mean_change=max(abs(a['mean_gap_m']-b['mean_gap_m']) for a,b in zip(own[-1]['futures'],reference['futures']))
            event_change=max(abs(a['separation_mass']-b['separation_mass']) for a,b in zip(own[-1]['futures'],reference['futures']))
            audits.append({'episode_id':episode['id'],'maximum_mean_difference_m':mean_change,
                'maximum_event_mass_difference':event_change,'adaptive_success':reference['success'],
                'log_integral_change':abs(reference['log_integral']-own[-1]['log_integral']),
                'empirical_tolerance_pass':bool(reference['success'] and mean_change<=settings['mean_change_tolerance_m']
                    and event_change<=settings['event_mass_change_tolerance']
                    and abs(reference['log_integral']-own[-1]['log_integral'])<=settings['log_integral_change_tolerance'])})
        if (index+1)%96==0:print(f'QUADRATURE {index+1}/{len(bundle["episodes"])} audits={len(audits)}',flush=True)
    write_json(root/'forecasts.json',rows);write_json(root/'refinement-checks.json',comparisons);write_json(root/'adaptive-checks.json',audits)
    names=('inputs.json','config.json','code-sha256.json','forecasts.json','refinement-checks.json','adaptive-checks.json')
    seals={n:digest(root/n) for n in names};write_json(root/'sealed-predictions.json',seals)
    return seals


def evaluate(root,bundle,settings,seals):
    # Synthetic labels are regenerated ONLY after predictions have been sealed.
    law=InterfaceLaw(**bundle['law']);fixture=Fixture(**bundle['source_settings']['fixture'])
    cases=new_cases(bundle['case_settings'],bundle['guard'],law,fixture,[])
    public={e['id']:e for e in bundle['episodes']};rows=[]
    last_checks={r['episode_id']:r for r in json.loads((root/'refinement-checks.json').read_text()) if r['to']==settings['primary_level']}
    predictions=json.loads((root/'forecasts.json').read_text())
    predictions += [{**r,'method':f'grid_{r["grid_side"]}','seconds':None} for r in bundle['baseline_forecasts']]
    truth_cache={}
    for prediction in predictions:
        episode=public[prediction['episode_id']];case=cases[int(episode['context_id'].split('-')[1])]
        if episode['id'] not in truth_cache:
            energy,preparation=case['parameters'];_,maximum=equilibrium_cycle(0.,energy,preparation,law,fixture)
            for action in episode['actions_m']:_,maximum=equilibrium_cycle(maximum,energy,action,law,fixture)
            futures=[]
            for action in bundle['source_settings']['future_drives_m']:
                gap,after=equilibrium_cycle(maximum,energy,action,law,fixture)
                futures.append((float(gap),bool(damage(after,energy,law)>=1-1e-12)))
            truth_cache[episode['id']]=futures
        truth=truth_cache[episode['id']]
        mse=float(np.mean([((p['mean_gap_m']-t[0])*1000)**2 for p,t in zip(prediction['futures'],truth)]))
        brier=float(np.mean([(p['separation_mass']-t[1])**2 for p,t in zip(prediction['futures'],truth)]))
        rows.append({'episode_id':episode['id'],'context_id':episode['context_id'],'trajectory_id':episode['trajectory_id'],
            'case_id':case['id'],'cluster':case['cluster'],
            **{k:episode[k] for k in ('policy','sigma_m','noise_condition','budget')},
            'method':prediction['method'],'future_mean_squared_error_mm2':mse,'separation_brier':brier,
            'integration_seconds':prediction['seconds'],'empirical_refinement_pass':last_checks[episode['id']]['empirical_tolerance_pass']})
    fields=('noise_condition','sigma_m','policy','budget','method');groups=[]
    for key in sorted({tuple(r[k] for k in fields) for r in rows}):
        chosen=[r for r in rows if tuple(r[k] for k in fields)==key]
        times=[r['integration_seconds'] for r in chosen if r['integration_seconds'] is not None]
        groups.append({**dict(zip(fields,key)),'checkpoints':len(chosen),
            'gap_rmse_mm':float(np.sqrt(np.mean([r['future_mean_squared_error_mm2'] for r in chosen]))),
            'mean_separation_brier':float(np.mean([r['separation_brier'] for r in chosen])),
            'empirical_refinement_failures':sum(not r['empirical_refinement_pass'] for r in chosen),
            'mean_integration_ms':float(np.mean(times)*1000) if times else None})
    write_json(root/'evaluation-rows.json',rows)
    assert seals=={n:digest(root/n) for n in seals}
    return groups,rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path);parser.add_argument('--config',type=Path);args=parser.parse_args()
    project=Path(os.environ['WM_PROJECT_ROOT']);settings=json.loads((args.config or project/'config/response-quadrature.json').read_text())
    if 'parent_config' in settings:
        parent=json.loads((project/settings['parent_config']).read_text());parent.update(settings);settings=parent
    stage='response-quadrature';engine_class=ResponseQuadrature
    if settings.get('quadrature_kind')=='guided':
        from wmfracture.guided_quadrature import GuidedResponseQuadrature
        stage='guided-quadrature';engine_class=GuidedResponseQuadrature
    source=None
    if args.inputs:bundle=json.loads(args.inputs.read_text())
    else:
        source=Path(json.loads((project/settings['source_report']).read_text())['run_directory']);bundle=bundle_from_source(source)
    root=Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+stage)
    root.mkdir(parents=True);print(f'OUTPUT={root}',flush=True)
    write_json(root/'config.json',settings);write_json(root/'inputs.json',bundle)
    paths=[Path(__file__),project/'wmfracture/response_quadrature.py',project/'wmfracture/cohesive.py',
        project/'wmfracture/compliant_fixture.py',project/'wmfracture/compliant_inference.py',
        project/'wmfracture/measurement_bias.py',project/'wmfracture/joint_state.py',
        project/'scripts/continuous_risk_evaluate.py',project/'scripts/history_refinement.py']
    if stage=='guided-quadrature':paths.append(project/'wmfracture/guided_quadrature.py')
    for p in paths:
        target=root/'source'/p.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in paths})
    start=time.perf_counter();seals=forecast_public(root,bundle,settings,engine_class=engine_class);forecast_seconds=time.perf_counter()-start
    groups,rows=evaluate(root,bundle,settings,seals)
    checks=json.loads((root/'refinement-checks.json').read_text());audits=json.loads((root/'adaptive-checks.json').read_text())
    last=[r for r in checks if r['to']==settings['primary_level']]
    if source:assert bundle['source_sha256']=={n:digest(source/n) for n in bundle['source_sha256']}
    report={'run_directory':str(root),'settings':settings,'source_run_id':bundle['source_run_id'],'groups':groups,
        'public_checkpoints':len(bundle['episodes']),'evaluation_rows':len(rows),'forecast_seconds':forecast_seconds,
        'total_seconds':time.perf_counter()-start,'refinement_failures':sum(not r['empirical_tolerance_pass'] for r in last),
        'maximum_last_mean_change_m':max(r['maximum_mean_change_m'] for r in last),
        'maximum_last_event_mass_change':max(r['maximum_event_mass_change'] for r in last),
        'adaptive_audits':len(audits),'adaptive_failed_checks':sum(not r['empirical_tolerance_pass'] for r in audits),
        'maximum_adaptive_mean_difference_m':max(r['maximum_mean_difference_m'] for r in audits),
        'original_inputs_and_forecasts_unchanged':True,'actions_or_policies_changed':False,
        'posterior_model_changed':False,'formally_certified_quadrature_error':False,'new_method_novelty_validated':False,
        'seals':seals,'scope':settings['scope']}
    write_json(root/'report.json',report);out=project/'artifacts'/stage;out.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','report.json','inputs.json','evaluation-rows.json','refinement-checks.json','adaptive-checks.json','code-sha256.json','sealed-predictions.json'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','settings','seals')},indent=2),flush=True)


if __name__=='__main__':main()
