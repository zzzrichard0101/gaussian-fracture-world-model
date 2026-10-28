"""Analytical finite-ensemble action diagnostics for the stable scalar fixture.

Standard bilinear-law inversion and piecewise-affine moment sweep. These do
not bound a continuous parameter domain or establish task information limits.
"""
import numpy as np

from wmfracture.compliant_inference import damage


def finite_action_limit(maximum, baseline, energy, cap, domain_max, law, fixture):
    """Largest admissible remote input for a NONEMPTY finite candidate set.

    Separation is excluded with a small numerical margin. If past cumulative
    damage already violates the cap, return zero and valid=False: no future
    input can undo it. Empty support is not interpreted as unrestricted safety.
    """
    maximum,baseline,energy = np.asarray(maximum,float),np.asarray(baseline,float),np.asarray(energy,float)
    if (maximum.ndim != 1 or maximum.shape != baseline.shape or maximum.shape != energy.shape
            or not np.isfinite([maximum,baseline,energy]).all() or not np.isfinite([cap,domain_max]).all()
            or cap < 0 or domain_max < 0 or (baseline < 0).any() or (maximum < baseline-1e-14).any()):
        raise ValueError('Invalid finite action limit')
    if not len(energy):
        return 0.,False
    final = 2*energy/law.peak_traction
    onset = law.initiation_opening
    if (energy <= 0).any() or (final <= onset).any() or (fixture.spring <= law.area*law.peak_traction/(final-onset)).any():
        raise ValueError('Stable positive cohesive branches required')
    initial,current = damage(baseline,energy,law),damage(maximum,energy,law)
    if np.any(current-initial > cap+1e-12) or np.any((current >= 1-1e-12) & (initial < 1-1e-12)):
        return 0.,False
    target = np.minimum(initial+cap,1-1e-9)
    kappa = final*onset/(final-target*(final-onset))
    limits = kappa*(1+law.area*law.tensile_stiffness*(1-target)/fixture.spring)
    limits = np.where(initial >= 1-1e-12,domain_max,limits)
    return float(max(0.,min(domain_max,float(limits.min())))*(1-1e-12)),True


def proxy_optimal_action(prefix,observed,sigma,tau,weights,maximum,energy,domain_max,law,fixture):
    """Globally maximize the existing variance proxy on [0,domain_max].

    Each candidate's next observed conditional mean is piecewise affine in
    nonnegative input. Its ensemble variance is convex quadratic between all
    mode switches, so a maximum occurs at an endpoint or a mode switch. A
    sorted moment sweep evaluates those candidates without an N-by-N rollout.
    This is NOT exact mutual information, Bayes utility or future error gain.
    """
    prefix,observed = np.asarray(prefix,float),np.asarray(observed,float)
    maximum,energy,weights = np.asarray(maximum,float),np.asarray(energy,float),np.asarray(weights,float)
    if (prefix.ndim != 2 or not prefix.shape[1] or observed.shape != (prefix.shape[1],)
            or maximum.shape != (len(prefix),) or energy.shape != maximum.shape or weights.shape != maximum.shape
            or not np.isfinite(prefix).all() or not np.isfinite(observed).all()
            or not np.isfinite([maximum,energy,weights]).all() or (maximum < 0).any() or (weights < 0).any()
            or weights.sum() <= 0 or not np.isfinite([sigma,tau,domain_max]).all()
            or sigma <= 0 or tau < 0 or domain_max < 0):
        raise ValueError('Invalid finite-ensemble proxy optimization')
    keep = weights > 0
    maximum,energy,weights,prefix = maximum[keep],energy[keep],weights[keep],prefix[keep]
    weights = weights/weights.sum()
    if len(weights)==1:
        return {'action_m':0.,'score':0.,'mode_switches':0,'weighted_candidates':len(weights),'direct_check_queries':0}
    if domain_max==0:
        from wmfracture.damage_guard import information_scores
        score = information_scores(prefix,observed,sigma,tau,weights,maximum,energy,[0.],law,fixture)[0]
        return {'action_m':0.,'score':float(score),'mode_switches':0,'weighted_candidates':len(weights),'direct_check_queries':len(weights)}
    final,onset = 2*energy/law.peak_traction,law.initiation_opening
    soft = law.area*law.peak_traction/(final-onset)
    if (final <= onset).any() or (fixture.spring <= soft).any():
        raise ValueError('Stable branches required for affine sweep')
    current = damage(maximum,energy,law)
    bias = tau**2*np.sum(observed-prefix,axis=1)/(sigma**2+prefix.shape[1]*tau**2)
    conditional_variance = tau**2*sigma**2/(sigma**2+prefix.shape[1]*tau**2)
    slope = fixture.spring/(fixture.spring+law.area*law.tensile_stiffness*(1-current))
    intercept = bias.copy()
    initial = np.array([weights@slope,weights@intercept,weights@(slope*slope),
        weights@(slope*intercept),weights@(intercept*intercept)])
    unbroken = current < 1-1e-12
    first = np.maximum(maximum,onset)/slope
    soft_slope = fixture.spring/(fixture.spring-soft)
    soft_intercept = -soft*final/(fixture.spring-soft)+bias
    def change(old_s,old_c,new_s,new_c):
        return np.column_stack((new_s-old_s,new_c-old_c,new_s*new_s-old_s*old_s,
            new_s*new_c-old_s*old_c,new_c*new_c-old_c*old_c))*weights[:,None]
    changes = np.concatenate((change(slope,intercept,soft_slope,soft_intercept),
        change(soft_slope,soft_intercept,np.ones(len(weights)),bias)),axis=0)
    events = np.r_[first,final]
    include = np.r_[unbroken,unbroken] & (events > 0) & (events <= domain_max)
    events,changes = events[include],changes[include]
    order = np.argsort(events,kind='stable');events,changes = events[order],changes[order]
    moments = initial[None,:]+np.cumsum(changes,axis=0)
    final_moments = moments[-1] if len(events) else initial
    points = np.r_[0.,events,domain_max]
    all_moments = np.vstack((initial,moments,final_moments))
    mean = all_moments[:,0]*points+all_moments[:,1]
    variances = np.maximum(0.,all_moments[:,2]*points**2+2*all_moments[:,3]*points+all_moments[:,4]-mean**2)
    best = int(np.argmax(variances))
    # Direct evaluation at the selected point checks cancellation in the sweep.
    from wmfracture.damage_guard import information_scores
    score = information_scores(prefix,observed,sigma,tau,weights,maximum,energy,[points[best]],law,fixture)[0]
    return {'action_m':float(points[best]),'score':float(score),'mode_switches':len(events),
        'weighted_candidates':len(weights),'direct_check_queries':len(weights)}


def noiseless_history_manifold(ratio,energy_bounds,preparation_bounds,weak_drive,cap,domain_max,law,fixture):
    """Restricted exact, noise-free ONE-STEP diagnostic for a damaged interface.

    Known unloading stiffness ratio q fixes D=1-q, but compatible energy/history
    pairs form kappa(E)=delta0*delta_f/(delta0+q*(delta_f-delta0)). This increases
    with E. Preparation bounds and weak loading below kappa clip the E interval.
    The cap-limited drive also increases with E, so its minimum is attained at
    the lowest compatible E. This result does not cover noisy 2D observation
    sets, arbitrary dynamics, or impossibility of sequential experiments.
    """
    values = np.r_[ratio,energy_bounds,preparation_bounds,weak_drive,cap,domain_max]
    if (not np.isfinite(values).all() or not 0 < ratio < 1 or not 0 < energy_bounds[0] < energy_bounds[1]
            or not 0 <= preparation_bounds[0] < preparation_bounds[1] or weak_drive < 0 or cap < 0 or domain_max < 0):
        raise ValueError('Invalid exact damaged-branch diagnostic')
    onset = law.initiation_opening
    scale = 1+law.area*law.tensile_stiffness*ratio/fixture.spring
    weak_gap = weak_drive/scale
    lower_kappa = max(onset,weak_gap,preparation_bounds[0]/scale)
    upper_kappa = preparation_bounds[1]/scale
    asymptote = onset/ratio
    if lower_kappa >= asymptote or lower_kappa > upper_kappa:
        return {'valid':False,'scope':'No compatible damaged unloading state'}
    def energy_from_kappa(kappa):
        return law.peak_traction*.5*kappa*onset*(1-ratio)/(onset-ratio*kappa)
    low = max(energy_bounds[0],energy_from_kappa(lower_kappa))
    high = min(energy_bounds[1],energy_from_kappa(upper_kappa) if upper_kappa < asymptote else energy_bounds[1])
    if low > high:
        return {'valid':False,'scope':'No compatible energy interval'}
    final = 2*low/law.peak_traction
    kappa = onset*final/(onset+ratio*(final-onset))
    limit,valid = finite_action_limit(np.array([kappa]),np.array([kappa]),np.array([low]),cap,domain_max,law,fixture)
    return {'valid':valid,'energy_interval':[float(low),float(high)],'weak_gap_m':weak_gap,
        'limiting_history_m':float(kappa),'one_step_limit_m':limit,
        'scope':'Exact damaged unloading manifold, known bias/noiseless initial stiffness, stable scalar law; numerical no-separation margin. Not a noisy-domain certificate or multi-step information bound.'}
