"""One-step Bayesian task design over a stable physical quadrature belief.

The Gaussian observation proxy is unchanged. Every executed action needs a
separate exact-rational risk proof; acquisition quadrature is not a safety bound.
"""
from dataclasses import asdict
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

from wmfracture.guided_quadrature import GuidedResponseQuadrature
from wmfracture.compliant_inference import PrefixModel,grid
from wmfracture.damage_guard import feasible_bias_interval,action_risks
from wmfracture.action_ceiling import finite_action_limit
from wmfracture.verified_policy import RiskReuseSession,filter_proposal


def risk_inputs(prefix,sigma,guard,law,fixture):
    return {'model_kind':'stable_quasistatic_nonnegative_cycles','parameter_bounds':guard['parameter_bounds'],
        'spring':fixture.spring,'law':{k:asdict(law)[k] for k in ('area','tensile_stiffness','peak_traction')},
        'actions_m':list(prefix['actions_m']),'observed_gaps_m':list(prefix['observed_gaps_m']),
        'reference_count':2,'noise_radius_m':float(guard['noise_multiple']*sigma),
        'bias_bound_m':guard['bias_bound_m'],'damage_cap':guard['damage_cap']}


def stable_belief(prefix,sigma,guard,law,fixture,settings):
    model=GuidedResponseQuadrature(law,fixture,guard['parameter_bounds'],prefix['actions_m'],prefix['observed_gaps_m'],sigma,guard['bias_sigma_m'])
    commands=settings['future_drives_m'];previous=None;checks=[];last=None
    for level in settings['belief_levels']:
        last=model.integrate(commands,**level);forecast=model.forecast(last,commands)
        if previous is not None:
            a,b=previous
            mean=max(abs(x['mean_gap_m']-y['mean_gap_m']) for x,y in zip(a,forecast))
            event=max(abs(x['separation_mass']-y['separation_mass']) for x,y in zip(a,forecast))
            logchange=abs(b-last.log_integral)
            passed=bool(mean<=settings['mean_change_tolerance_m'] and event<=settings['event_change_tolerance']
                and logchange<=settings['log_integral_change_tolerance'])
            checks.append({'mean_change_m':mean,'event_mass_change':event,'log_integral_change':logchange,'passed':passed})
            if passed:return model,last,True,checks
        previous=(forecast,last.log_integral)
    return model,last,False,checks


def compress_mixture(means,weights,targets,noise_std,bin_std,max_bins):
    means,weights,targets=np.asarray(means,float),np.asarray(weights,float),np.asarray(targets,float)
    if (means.ndim!=1 or weights.shape!=means.shape or targets.ndim!=2 or len(targets)!=len(means)
            or not np.isfinite(means).all() or not np.isfinite(weights).all() or not np.isfinite(targets).all()
            or (weights<0).any() or weights.sum()<=0 or noise_std<=0 or bin_std<=0):
        raise ValueError('Finite predictive mixture and positive resolution required')
    weights=weights/weights.sum();origin=float(weights@means)
    keys=np.floor((means-origin)/(noise_std*bin_std)).astype(np.int64)
    _,inverse=np.unique(keys,return_inverse=True);count=int(inverse.max()+1)
    if count>max_bins:raise ArithmeticError('Acquisition mixture bin budget exceeded')
    mass=np.bincount(inverse,weights=weights)
    active=mass>0;mass=mass[active]
    centers=np.bincount(inverse,weights=weights*means)[active]/mass
    values=np.column_stack([np.bincount(inverse,weights=weights*targets[:,i])[active]/mass for i in range(targets.shape[1])])
    return centers,mass,values


def expected_task_gain(means,weights,targets,noise_variance,bin_std=.125,outcome_order=12,max_bins=4096,max_panels=512):
    """E_z ||E[F|z]-E[F]||^2, averaged over target future commands.

    F is in mm, hence gain is mm^2. Moment-preserving mixture compression is
    refined separately from the outcome quadrature; neither is a risk proof.
    """
    if not np.isfinite(noise_variance) or noise_variance<=0:raise ValueError('Positive predictive noise variance required')
    weights=np.asarray(weights,float);weights=weights/weights.sum();targets=np.asarray(targets,float)
    center=weights@targets;prior=float(np.mean(weights@((targets-center)**2)))
    std=np.sqrt(noise_variance)
    mu,mass,feature=compress_mixture(means,weights,targets,std,bin_std,max_bins)
    origin=float(mass@mu);locations=(mu-origin)/std
    lo,hi=float(locations.min()-8),float(locations.max()+8)
    panels=max(1,int(np.ceil((hi-lo)/2)))
    if panels>max_panels:raise ArithmeticError('Acquisition outcome panel budget exceeded')
    cuts=np.linspace(lo,hi,panels+1);nodes,gw=leggauss(outcome_order)
    x=np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in zip(cuts[:-1],cuts[1:])])
    quadrature=np.concatenate([(b-a)/2*gw for a,b in zip(cuts[:-1],cuts[1:])])
    density=np.exp(-.5*(x[:,None]-locations[None,:])**2)/np.sqrt(2*np.pi)
    marginal=density@mass
    numerator=density@(mass[:,None]*(feature-center))
    value=np.mean(numerator**2/np.maximum(marginal[:,None],np.finfo(float).tiny),axis=1)
    gain=float(quadrature@value);total_mass=float(quadrature@marginal)
    valid=bool(np.isfinite(gain) and gain>=0 and gain<=prior+1e-9*max(prior,1.) and abs(total_mass-1)<1e-7)
    return {'gain_mm2':min(gain,prior) if valid else None,'prior_variance_mm2':prior,
        'expected_posterior_variance_mm2':max(prior-gain,0.) if valid else None,
        'predictive_mass':total_mass,'bins':len(mu),'outcome_nodes':len(x),'valid':valid,
        'scope':'Gaussian-proxy one-step expected variance reduction with numerical mixture/outcome approximations.'}


class PredictiveMixture:
    def __init__(self,model,result,settings):
        if min(settings['future_drives_m'])<max(model.bounds[1,1],settings['maximum_drive_m'],max(model.actions)):
            raise ValueError('This task requires future commands above possible preparation/probe peaks')
        order=np.argsort(-result.weights,kind='stable');cumulative=np.cumsum(result.weights[order])
        count=min(len(order),int(np.searchsorted(cumulative,1-settings['posterior_trim_mass']))+1)
        keep=np.sort(order[:count]);self.omitted_mass=max(0.,float(1-result.weights[keep].sum()))
        self.model=model;self.parameters=result.parameters[keep];self.weights=result.weights[keep].copy()
        self.weights/=self.weights.sum()
        prediction=model.predict(self.parameters,model.actions);n=len(model.actions)
        self.bias_mean=model.tau**2*np.sum(model.observed-prediction,axis=1)/(model.sigma**2+n*model.tau**2)
        self.noise_variance=float(model.sigma**2+model.tau**2*model.sigma**2/(model.sigma**2+n*model.tau**2))
        energy,history=self.parameters.T
        self.targets=np.column_stack([a*model.compliance(energy,max(a,max(model.actions)))*1000 for a in settings['future_drives_m']])

    def means(self,action):
        energy,history=self.parameters.T
        return action*self.model.compliance(energy,np.maximum(history,max(action,max(self.model.actions))))+self.bias_mean

    def disagreement(self,action):
        mu=self.means(action);center=float(self.weights@mu)
        return float(.5*np.log1p((self.weights@((mu-center)**2))/self.noise_variance))

    def task_score(self,action,settings):
        mu=self.means(action);previous=None;records=[]
        for level in settings['acquisition_levels']:
            try:
                result=expected_task_gain(mu,self.weights,self.targets,self.noise_variance,
                    **level,max_bins=settings['acquisition_max_bins'],max_panels=settings['acquisition_max_panels'])
            except ArithmeticError as exc:
                return None,{'stable':False,'reason':str(exc),'levels':records}
            records.append(result)
            if previous is not None and result['valid'] and previous['valid']:
                difference=abs(result['gain_mm2']-previous['gain_mm2'])
                tolerance=settings['gain_absolute_tolerance_mm2']+settings['gain_relative_tolerance']*result['prior_variance_mm2']
                if difference<=tolerance:
                    return result['gain_mm2'],{'stable':True,'difference_mm2':difference,'tolerance_mm2':tolerance,'levels':records}
            previous=result
        return None,{'stable':False,'reason':'Acquisition refinement did not pass','levels':records}


class PublicRiskBank:
    def __init__(self,inputs,guard,law,fixture):
        self.inputs,self.guard,self.law,self.fixture=inputs,guard,law,fixture
        self.model=PrefixModel(law,fixture,guard['parameter_bounds']);self.unit=grid(guard['grid_side'])
        self.parameters=self.model.parameters(self.unit)
        prediction,self.maximum=self.model.predict(self.unit,inputs['actions_m'])
        _,self.baseline=self.model.predict(self.unit,inputs['actions_m'][:2])
        self.mask,_,_=feasible_bias_interval(prediction,inputs['observed_gaps_m'],inputs['noise_radius_m'],1.,inputs['bias_bound_m'])

    def warm(self,action):
        indices=np.flatnonzero(self.mask)
        if not len(indices):return []
        risks,_=action_risks(self.maximum[indices],self.baseline[indices],self.parameters[indices,0],[action],self.law,self.fixture)
        selected=indices[np.argsort(-risks[0],kind='stable')[:5]]
        return self.parameters[selected].tolist()

    def limit(self,maximum):
        return finite_action_limit(self.maximum[self.mask],self.baseline[self.mask],self.parameters[self.mask,0],
            self.guard['damage_cap'],maximum,self.law,self.fixture)


class TaskController:
    """Stable belief -> certified candidate inputs -> public one-step scores."""
    def __init__(self,policy,law,fixture,guard,settings,sigma):
        if policy not in ('observation_disagreement','future_task'):raise ValueError('Unknown adaptive policy')
        self.policy,self.law,self.fixture,self.guard,self.settings,self.sigma=policy,law,fixture,guard,settings,sigma
        self.session=RiskReuseSession()

    def plan(self,prefix):
        tick=time.perf_counter();s=self.settings
        model,result,stable,checks=stable_belief(prefix,self.sigma,self.guard,self.law,self.fixture,s)
        belief_seconds=time.perf_counter()-tick
        inputs=risk_inputs(prefix,self.sigma,self.guard,self.law,self.fixture)
        bank=PublicRiskBank(inputs,self.guard,self.law,self.fixture)
        limit,valid=bank.limit(s['maximum_drive_m']);proposal=limit if valid else s['maximum_drive_m']
        frontier=filter_proposal(inputs,proposal,bank.warm,max_cells=s['cell_budget'],
            backtracking_steps=s['backtracking_steps'],reuse_session=self.session)
        record={'belief_stable':stable,'belief_checks':checks,'belief_seconds':belief_seconds,
            'belief_nodes':result.diagnostics,'frontier_proposal_m':proposal,'frontier_attempts':frontier['attempts'],
            'candidate_checks':[],'scores':[],'acquisition_abstained':False}
        if frontier['stopped']:return None,None,record
        ceiling=frontier['action_m'];pool={ceiling:frontier['proof']}
        requested=sorted(set([0.,self.guard['weak_drive_m'],min(max(prefix['actions_m']),ceiling)]+
            [ceiling*f for f in s['candidate_fractions']]))
        for action in requested:
            if action in pool:continue
            filtered=filter_proposal(inputs,action,bank.warm,max_cells=s['cell_budget'],backtracking_steps=0,reuse_session=self.session)
            record['candidate_checks'].append({'requested_m':action,'actual_m':filtered['action_m'],'attempts':filtered['attempts']})
            if not filtered['stopped']:pool[filtered['action_m']]=filtered['proof']
        if not stable:
            record['acquisition_abstained']=True;record['reason']='Belief refinement failed'
            return (0.,pool[0.],record) if 0. in pool else (None,None,record)
        mixture=PredictiveMixture(model,result,s);record['posterior_omitted_mass']=mixture.omitted_mass
        record['weighted_nodes']=len(mixture.weights);score_start=time.perf_counter();failed=False
        for action in sorted(pool):
            if self.policy=='observation_disagreement':score=mixture.disagreement(action);details={'stable':True}
            else:score,details=mixture.task_score(action,s);failed=failed or score is None
            record['scores'].append({'action_m':action,'score':score,'details':details})
        record['score_seconds']=time.perf_counter()-score_start
        if failed:
            record['acquisition_abstained']=True;record['reason']='Acquisition refinement/budget failed'
            return (0.,pool[0.],record) if 0. in pool else (None,None,record)
        ordered=sorted(record['scores'],key=lambda r:(-r['score'],r['action_m']))
        best=ordered[0];minimum=s['minimum_gain_mm2'] if self.policy=='future_task' else 1e-8
        if best['score']<=minimum and 0. in pool:best=next(r for r in ordered if r['action_m']==0.)
        # Ties at the declared acquisition tolerance favor the smaller input.
        if self.policy=='future_task':
            tolerance=2*max((r['details'].get('difference_mm2',0.) for r in ordered),default=0.)+1e-12
            near=[r for r in ordered if best['score']-r['score']<=tolerance]
            record['near_tied_candidates']=len(near)
            record['empirical_rank_tolerance_mm2']=tolerance
            best=min(near,key=lambda r:r['action_m'])
        record['selected_score']=best['score'];record['ceiling_m']=ceiling
        return best['action_m'],pool[best['action_m']],record
