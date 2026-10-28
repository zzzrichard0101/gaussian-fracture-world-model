"""Finite-candidate cumulative-damage checks; NOT continuous safety bounds.

The observation set assumes a bounded persistent bias and bounded errors.
Gaussian samples in experiments can violate those bounds. Grid containment,
true-state containment and model correctness are separate assumptions.
"""
import numpy as np

from wmfracture.compliant_inference import equilibrium_cycle, damage, normalize, grid, PrefixModel
from wmfracture.measurement_bias import GaussianObservation


def feasible_bias_interval(predictions, observed, sigma, noise_multiple, bias_bound):
    predictions, observed = np.asarray(predictions,float), np.asarray(observed,float)
    if (predictions.ndim != 2 or not predictions.shape[1] or observed.shape != (predictions.shape[1],)
            or not np.isfinite(predictions).all() or not np.isfinite(observed).all()
            or not np.isfinite([sigma,noise_multiple,bias_bound]).all()
            or sigma <= 0 or noise_multiple <= 0 or bias_bound < 0):
        raise ValueError('Invalid bounded-error observation set')
    residual = observed-predictions
    radius = noise_multiple*sigma
    lower = np.maximum(np.max(residual-radius,axis=1),-bias_bound)
    upper = np.minimum(np.min(residual+radius,axis=1),bias_bound)
    return lower <= upper, lower, upper


def action_risks(maximum, baseline, energy, actions, law, fixture):
    """Cumulative extra D relative to a FIXED initial-observation reference."""
    maximum, baseline, energy = np.asarray(maximum,float), np.asarray(baseline,float), np.asarray(energy,float)
    actions = np.asarray(actions,float)
    if (maximum.ndim != 1 or maximum.shape != baseline.shape or energy.shape != maximum.shape
            or not np.isfinite([maximum,baseline,energy]).all() or (baseline < 0).any()
            or (maximum < baseline-1e-14).any() or actions.ndim != 1 or not len(actions)
            or not np.isfinite(actions).all() or (actions < 0).any()):
        raise ValueError('Invalid fixed-reference risk query')
    _, after = equilibrium_cycle(maximum[None,:],energy[None,:],actions[:,None],law,fixture)
    reference_damage = damage(baseline,energy,law)
    values = damage(after,energy[None,:],law)-reference_damage[None,:]
    new_failures = (damage(after,energy[None,:],law) >= 1-1e-12) & (reference_damage[None,:] < 1-1e-12)
    return np.maximum(values,0.),new_failures


def admissibility(risks, failures, cap):
    risks, failures = np.asarray(risks,float), np.asarray(failures,bool)
    if risks.ndim != 2 or risks.shape != failures.shape or not np.isfinite(risks).all() or not np.isfinite(cap) or cap < 0:
        raise ValueError('Invalid finite risk bank')
    if not risks.shape[1]:
        return np.zeros(risks.shape[0],dtype=bool)
    return (np.max(risks,axis=1) <= cap+1e-12) & ~np.any(failures,axis=1)


def top_weight_indices(weights, mask, count):
    weights, mask = np.asarray(weights,float), np.asarray(mask,bool)
    if weights.ndim != 1 or weights.shape != mask.shape or not np.isfinite(weights).all() or (weights < 0).any() or count < 1:
        raise ValueError('Invalid weighted subset')
    indices = np.flatnonzero(mask)
    return indices[np.argsort(-weights[indices],kind='stable')[:count]]


def risk_witnesses(risks, failures, cap):
    """One offending sample per forbidden DESIGN action; full scan is required.

    Preserves admissibility on that finite action bank, not all risk maxima,
    unseen actions, future observations or a continuous parameter domain.
    """
    allowed = admissibility(risks,failures,cap)
    witnesses = set()
    for action in np.flatnonzero(~allowed):
        if risks.shape[1] == 0:
            continue
        if failures[action].any():
            witnesses.add(int(np.flatnonzero(failures[action])[0]))
        else:
            witnesses.add(int(np.argmax(risks[action])))
    return np.asarray(sorted(witnesses),dtype=int)


def information_scores(prefix, observed, sigma, tau, weights, maximum, energy, actions, law, fixture):
    """Moment-based observation-disagreement proxy, not exact mutual information.

    Uses the CONDITIONAL offset mean for each physical hypothesis, rather than
    treating a future shared-bias observation as independent of past ones.
    """
    weights = np.asarray(weights,float)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Positive information-ensemble weight required')
    weights = weights/total
    n = prefix.shape[1]
    bias_mean = tau**2*np.sum(np.asarray(observed)-prefix,axis=1)/(sigma**2+n*tau**2)
    conditional_variance = tau**2*sigma**2/(sigma**2+n*tau**2)
    gap,_ = equilibrium_cycle(maximum[None,:],energy[None,:],np.asarray(actions)[:,None],law,fixture)
    mean = gap+bias_mean[None,:]
    center = np.sum(mean*weights[None,:],axis=1)
    variance = np.sum(weights[None,:]*(mean-center[:,None])**2,axis=1)
    return .5*np.log1p(variance/(sigma**2+conditional_variance))


POLICIES = ('rest_only','fixed_probe','top8_guard','full_guard','witness_guard','full_guard_info8','continuous_guard')


class FiniteGuard:
    """Controller gets public observations and known mechanics, never true state."""
    def __init__(self, law, fixture, settings, sigma):
        self.law,self.fixture,self.settings,self.sigma = law,fixture,settings,sigma
        self.model = PrefixModel(law,fixture,settings['parameter_bounds'])
        self.unit = grid(settings['grid_side']);self.energy = self.model.parameters(self.unit)[:,0]
        self.actions = np.asarray(settings['actions_m'],float)
        if self.actions.ndim != 1 or len(set(self.actions)) != len(self.actions) or self.actions[0] != 0 or (np.diff(self.actions) <= 0).any():
            raise ValueError('Guard actions must be sorted, unique and start at zero')
        self.design_actions = self.actions.copy()
        self.reference_maximum = None
        self.previous_actions,self.previous_observed = [],[]
        self.previous_mask = np.ones(len(self.unit),dtype=bool)

    def plan(self, actions, observed, policy, round_index):
        if policy not in POLICIES:
            raise ValueError('Unknown diagnostic policy')
        actions,observed = list(actions),np.asarray(observed,float)
        previous = len(self.previous_actions)
        if (len(actions) != len(observed) or actions[:previous] != self.previous_actions
                or not np.array_equal(observed[:previous],self.previous_observed)
                or actions[:2] != [0.,self.settings['weak_drive_m']]
                or len(actions) != 2+round_index):
            raise ValueError('Controller must continue its initial zero/weak observation prefix')
        prediction,maximum = self.model.predict(self.unit,actions)
        if self.reference_maximum is None:
            if round_index != 0:
                raise ValueError('Missing initial damage reference')
            self.reference_maximum = maximum.copy()
        mask,lower,upper = feasible_bias_interval(prediction,observed,self.sigma,
            self.settings['noise_multiple'],self.settings['bias_bound_m'])
        if np.any(mask & ~self.previous_mask):
            raise AssertionError('Observation set grew under an unchanged parameter prefix')
        weights = normalize(GaussianObservation('shared',self.settings['bias_sigma_m'])(prediction,observed,self.sigma))
        self.actions = self.design_actions.copy()
        continuous = None
        if policy == 'continuous_guard':
            from wmfracture.action_ceiling import finite_action_limit,proxy_optimal_action
            limit,valid = finite_action_limit(maximum[mask],self.reference_maximum[mask],self.energy[mask],
                self.settings['damage_cap'],float(self.design_actions[-1]),self.law,self.fixture)
            continuous = proxy_optimal_action(prediction,observed,self.sigma,self.settings['bias_sigma_m'],weights,
                maximum,self.energy,limit,self.law,self.fixture)
            continuous.update({'finite_limit_m':limit,'finite_limit_valid':valid})
            self.actions = np.unique(np.r_[self.design_actions,limit,continuous['action_m']])
        info_indices = (top_weight_indices(weights,np.ones(len(weights),bool),self.settings['subset_size'])
            if policy == 'full_guard_info8' else np.arange(len(weights)))
        if policy in ('rest_only','fixed_probe'):
            info_indices = np.empty(0,dtype=int)
            score = np.zeros(len(self.actions))
        else:
            score = information_scores(prediction[info_indices],observed,self.sigma,self.settings['bias_sigma_m'],
                weights[info_indices],maximum[info_indices],self.energy[info_indices],self.actions,self.law,self.fixture)
        feasible = np.flatnonzero(mask)
        selected = top_weight_indices(weights,mask,self.settings['subset_size'])
        planning_queries = len(info_indices)*len(self.actions)
        if continuous is not None:
            planning_queries += continuous['direct_check_queries']
        guard_queries,construction_queries,audit_queries = 0,0,0
        risks = failures = None
        witness_indices = np.empty(0,dtype=int)
        if policy in ('full_guard','full_guard_info8','witness_guard','continuous_guard'):
            risks,failures = action_risks(maximum[feasible],self.reference_maximum[feasible],self.energy[feasible],self.actions,self.law,self.fixture)
            if policy == 'witness_guard':
                construction_queries = risks.size
                witnesses = risk_witnesses(risks,failures,self.settings['damage_cap'])
                witness_indices = np.unique(np.r_[selected,feasible[witnesses]])
                limited_risk,limited_failure = action_risks(maximum[witness_indices],self.reference_maximum[witness_indices],self.energy[witness_indices],self.actions,self.law,self.fixture)
                allowed = admissibility(limited_risk,limited_failure,self.settings['damage_cap'])
                guard_queries = limited_risk.size
                assert np.array_equal(allowed,admissibility(risks,failures,self.settings['damage_cap']))
            else:
                allowed = admissibility(risks,failures,self.settings['damage_cap']);guard_queries = risks.size
        elif policy == 'top8_guard':
            limited_risk,limited_failure = action_risks(maximum[selected],self.reference_maximum[selected],self.energy[selected],self.actions,self.law,self.fixture)
            allowed = admissibility(limited_risk,limited_failure,self.settings['damage_cap']);guard_queries = limited_risk.size
        else:
            allowed = np.ones(len(self.actions),dtype=bool)
        if policy == 'rest_only':
            choice,reason = 0,'rest_observation_control'
        elif policy == 'fixed_probe':
            target = self.settings['fixed_probes_m'][round_index]
            choice = int(np.flatnonzero(self.actions == target)[0]);reason = 'unconstrained_fixed_probe'
        elif not mask.any():
            choice,reason = 0,'empty_observation_set_abstention'
        elif not allowed.any() or float(np.max(score[allowed])) <= self.settings['minimum_information_score']:
            choice,reason = 0,'no_admissible_informative_action'
        else:
            choice = int(np.argmax(np.where(allowed,score,-np.inf)));reason = 'disagreement_subject_to_guard'
        # The full audit is strictly after action selection. Its computations
        # are charged separately; top8's planner never receives this result.
        if risks is None:
            risks,failures = action_risks(maximum[feasible],self.reference_maximum[feasible],self.energy[feasible],self.actions,self.law,self.fixture)
            audit_queries = risks.size
        full_allowed = admissibility(risks,failures,self.settings['damage_cap'])
        self.previous_actions,self.previous_observed = actions.copy(),observed.copy()
        self.previous_mask = mask.copy()
        record = {'policy':policy,'round':round_index,'action_m':float(self.actions[choice]),'reason':reason,
            'action_bank_m':self.actions.tolist(),'continuous_action_diagnostic':continuous,
            'feasible_candidates':len(feasible),'subset_candidates':len(selected),'witness_union_size':len(witness_indices),
            'admissible_actions':allowed.tolist(),'full_admissible_actions':full_allowed.tolist(),
            'information_scores':score.tolist(),'empty_set':not bool(mask.any()),
            'finite_sample_maximum_damage_by_action':np.max(risks,axis=1).tolist() if len(feasible) else None,
            'chosen_action_finite_sample_forbidden':bool(not full_allowed[choice]) if len(feasible) else None,
            'optimistic_admissibility_count':int(np.sum(allowed & ~full_allowed)) if policy.endswith('guard') or policy=='full_guard_info8' else None,
            'cost':{**self.model.cost,'information_action_candidate_queries':planning_queries,
                'analytical_limit_candidate_checks':int(mask.sum()) if continuous is not None else 0,
                'proxy_moment_event_updates':continuous['mode_switches'] if continuous is not None else 0,
                'guard_action_candidate_queries':guard_queries,'witness_construction_queries':construction_queries,
                'post_selection_audit_queries':audit_queries,
                'prefix_role':'post_selection_diagnostic_only' if policy in ('rest_only','fixed_probe') else 'planning'},
            'set_scope':'Finite prior grid satisfying bounded shared-offset/noise constraints; no continuous risk bound.'}
        return float(self.actions[choice]),record
