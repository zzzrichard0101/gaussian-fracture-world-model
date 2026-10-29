"""Response-chart importance sampling on a predeclared development subset."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from wmfracture.spatial_observation import PhysicalCatalog
from wmfracture.spatial_transport import PublicForward, bounded_loglike, fit_proposals, summarize_importance
from spatial_observation_evaluate import POLICY_KEYS, write


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='config/spatial-transport.json')
    args=parser.parse_args(); project=Path(__file__).resolve().parents[1]
    source=project/'docs/results'; output=project/'artifacts/spatial-transport'; output.mkdir(parents=True,exist_ok=True)
    config=json.loads((project/args.config).read_text()); settings=json.loads((source/'spatial-observation-config.json').read_text())
    old=json.loads((source/'spatial-observation-evaluation-rows.json').read_text())
    selected={r['id']:r for r in old if r['case_kind']=='outside' and r['modality']=='endpoints' and r['policy']=='finite_guard_tilt'
        and r['bias_index']==0 and r['repeat']==0}
    public=json.loads((source/'spatial-observation-public-episodes.json').read_text())
    episodes=[e for e in public if e['id'] in selected]
    ps={k:settings[k] for k in POLICY_KEYS}; bounds=np.asarray(config['parameter_bounds'])
    write(output/'config.json',config);write(output/'inputs.json',episodes)
    sources=['wmfracture/spatial_transport.py','scripts/spatial_transport_evaluate.py','tests/test_spatial_transport.py',
        'wmfracture/spatial_observation.py','wmfracture/response_quadrature.py',args.config]
    write(output/'code-sha256.json',{p:hashlib.sha256((project/p).read_bytes()).hexdigest() for p in sources})
    write(output/'source-sha256.json',{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in
        ('spatial-observation-config.json','spatial-observation-public-episodes.json','spatial-observation-evaluation-rows.json')})
    predictions=[]; adaptation=[]; started=time.perf_counter(); max_power=max(config['nested_powers']); n=2**max_power
    for replica,(pilot_seed,seed) in enumerate(zip(config['pilot_seeds'],config['scramble_seeds'])):
        pilot_unit=qmc.Sobol(3,scramble=True,seed=pilot_seed).random_base2(config['pilot_power'])
        pilot_bank=PhysicalCatalog(bounds[:,0]+pilot_unit*np.ptp(bounds,axis=1),settings['actions_m'],settings['future_drives_m'],config['cycle_increments'])
        unit4=qmc.Sobol(4,scramble=True,seed=seed).random_base2(max_power)
        prior_unit=unit4[:,1:]
        prior_bank=PhysicalCatalog(bounds[:,0]+prior_unit*np.ptp(bounds,axis=1),settings['actions_m'],settings['future_drives_m'],config['cycle_increments'])
        for ei,episode in enumerate(episodes):
            prefix=episode['actions']; positions=episode['positions_m']; observed=episode['observed_m']; sigma=episode['sigma_m']
            pilot_prediction=np.stack([pilot_bank.observations(tuple(prefix[:i+1]),positions) for i in range(len(prefix))],axis=1)
            forward=PublicForward(bounds,settings['actions_m'],settings['future_drives_m'],positions,config['cycle_increments'])
            tick=time.perf_counter()
            proposals,audit=fit_proposals(forward,pilot_unit,pilot_prediction,prefix,observed,sigma,ps,config)
            adaptation.append({'id':episode['id'],'replica':replica,'pilot_seed':pilot_seed,'audit':audit,
                'adaptation_parameter_evaluations':forward.parameter_evaluations,
                'adaptation_endpoint_parameter_evaluations':forward.endpoint_parameter_evaluations,
                'seconds':time.perf_counter()-tick})
            print(f'SPATIAL_TRANSPORT replica={replica} episode={ei+1}/{len(episodes)} projected',flush=True)
            prior_prediction=np.stack([prior_bank.observations(tuple(prefix[:i+1]),positions) for i in range(len(prefix))],axis=1)
            prior_gaps,prior_events=prior_bank.future(tuple(prefix))
            prior_loglike=bounded_loglike(prior_prediction,observed,sigma,ps)
            for method in config['methods']:
                tick=time.perf_counter(); before_count=forward.parameter_evaluations
                before_endpoints=forward.endpoint_parameter_evaluations
                outside_count=0; uniform_count=n
                if method=='prior_qmc':
                    loglike=prior_loglike; logq=np.zeros(n); gaps,events=prior_gaps,prior_events
                else:
                    points,logq=proposals[method].sample(unit4)
                    inside=np.all((points>=0.)&(points<=1.),axis=1); outside_count=int((~inside).sum())
                    prior_mask=unit4[:,0]<config['uniform_fraction']; uniform_count=int(prior_mask.sum())
                    active=inside&~prior_mask
                    prediction=np.zeros_like(prior_prediction);gaps=np.zeros_like(prior_gaps);events=np.zeros_like(prior_events)
                    prediction[prior_mask]=prior_prediction[prior_mask]
                    gaps[prior_mask]=prior_gaps[prior_mask];events[prior_mask]=prior_events[prior_mask]
                    if active.any():
                        fitted,future=forward.evaluate(points[active],prefix,future=True)
                        prediction[active]=fitted;gaps[active],events[active]=future
                    loglike=np.full(n,-np.inf)
                    if inside.any():loglike[inside]=bounded_loglike(prediction[inside],observed,sigma,ps)
                for power in config['nested_powers']:
                    count=2**power
                    result=summarize_importance(loglike[:count],logq[:count],gaps[:count],events[:count])
                    predictions.append({'id':episode['id'],'sigma_m':sigma,'method':method,'replica':replica,'seed':seed,'power':power,**result})
                adaptation.append({'id':episode['id'],'replica':replica,'method':method,
                    'main_parameter_evaluations':forward.parameter_evaluations-before_count,
                    'main_endpoint_parameter_evaluations':forward.endpoint_parameter_evaluations-before_endpoints,
                    'outside_cube_draws_at_maximum':outside_count,'cached_uniform_nodes_at_maximum':uniform_count,
                    'seconds_with_shared_uniform_cache':time.perf_counter()-tick})
                print(f'SPATIAL_TRANSPORT replica={replica} episode={ei+1}/{len(episodes)} {method}',flush=True)
    write(output/'predictions.json',predictions);write(output/'adaptation.json',adaptation)
    frozen=hashlib.sha256((output/'predictions.json').read_bytes()).hexdigest()
    # Actual physical parameters/futures are read only after estimates freeze.
    truth_parameters=json.loads((source/'spatial-observation-case-config.json').read_text())['true_parameters']
    truth=PhysicalCatalog(truth_parameters,settings['actions_m'],settings['future_drives_m'],config['cycle_increments'])
    look={(r['id'],r['method'],r['replica'],r['power']):r for r in predictions}
    previous=sorted(config['nested_powers'])[-2]; evaluated=[]
    for episode in episodes:
        ci=selected[episode['id']]['case_index'];targets,_=truth.future(tuple(episode['actions']));target=targets[ci]
        for method in config['methods']:
            rows=[look[(episode['id'],method,replica,power)] for replica in range(len(config['scramble_seeds'])) for power in (previous,max_power)]
            available=all(r['available'] for r in rows); mean_range=event_range=None
            if available:
                mean_range=float(np.max(np.ptp(np.asarray([r['mean_gap_m'] for r in rows]),axis=0)))
                event_range=float(np.max(np.ptp(np.asarray([r['event_probability'] for r in rows]),axis=0)))
            passed=bool(available and min(r['ess'] for r in rows)>=config['screen']['minimum_ess']
                and mean_range<=config['screen']['maximum_mean_change_m'] and event_range<=config['screen']['maximum_event_change'])
            primary=look[(episode['id'],method,0,max_power)]
            evaluated.append({'id':episode['id'],'sigma_m':episode['sigma_m'],'method':method,
                'available':primary['available'],'primary_ess':primary['ess'],'screen_passed':passed,
                'max_mean_range_m':mean_range,'max_event_range':event_range,
                'primary_mse_mm2':float(np.mean(((np.asarray(primary['mean_gap_m'])-target)*1000)**2)) if primary['available'] else None})
    summaries=[]
    for sigma in settings['sigma_m']:
        for method in config['methods']:
            group=[r for r in evaluated if r['sigma_m']==sigma and r['method']==method]
            valid=[r for r in group if r['available']];stable=[r for r in group if r['screen_passed']]
            summaries.append({'sigma_m':sigma,'method':method,'histories':len(group),'available':len(valid),
                'screen_passed':len(stable),'median_ess':float(np.median([r['primary_ess'] for r in group])),
                'rmse_mm_on_available':float(np.sqrt(np.mean([r['primary_mse_mm2'] for r in valid]))) if valid else None,
                'rmse_mm_on_screened':float(np.sqrt(np.mean([r['primary_mse_mm2'] for r in stable]))) if stable else None})
    write(output/'evaluation-rows.json',evaluated);write(output/'analysis-summary.json',summaries)
    report={'config':config,'histories':len(episodes),'prediction_rows':len(predictions),'summaries':summaries,
        'frozen_prediction_sha256':frozen,'elapsed_seconds':time.perf_counter()-started,
        'scope':config['scope'],'cost_scope':'Pilot/prior physics caches are shared. Adaptation and main node counts are separate; final-node equality is not matched total computation. No end-to-end speedup claim.'}
    write(output/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
