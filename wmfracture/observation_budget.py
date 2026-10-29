"""Public observation-budget policies and bounded synthetic noise generation."""
import numpy as np
from scipy.special import ndtr, ndtri

from wmfracture.damage_guard import FiniteGuard


POLICIES=('rest_repeat','weak_repeat','fixed_ramp','adaptive_verified')


def bounded_noise(seed, count, sigma, radius_multiple, rho=0.):
    """Stationary Gaussian AR(1) copula with truncated-normal marginals.

    rho is LATENT correlation, not the observed transformed Pearson correlation.
    Every marginal has the same truncated normal law for rho=0 and rho>0.
    Temporal correlation never changes the verifier's deterministic error bound.
    """
    if (type(count) is not int or count<1 or not np.isfinite([sigma,radius_multiple,rho]).all()
            or sigma<=0 or radius_multiple<=0 or not 0<=rho<1):
        raise ValueError('Positive count, noise scale/radius and stationary nonnegative rho required')
    innovations=np.random.default_rng(seed).normal(size=count)
    latent=innovations.copy()
    for index in range(1,count):
        latent[index]=rho*latent[index-1]+np.sqrt(1-rho*rho)*innovations[index]
    lower,upper=ndtr(-radius_multiple),ndtr(radius_multiple)
    probability=lower+(upper-lower)*ndtr(latent)
    # Guard inverse-CDF endpoint roundoff; the declared radius still holds.
    noise=np.clip(ndtri(probability),-radius_multiple,radius_multiple)*sigma
    return noise


class BudgetProposal:
    """Budget-blind proposal: no true state, future labels or intended horizon."""
    def __init__(self,policy,settings,guard,law,fixture,sigma):
        if policy not in POLICIES:raise ValueError('Unknown observation-budget policy')
        self.policy,self.settings,self.guard=policy,settings,guard
        self.controller=FiniteGuard(law,fixture,guard,sigma) if policy=='adaptive_verified' else None

    def plan(self,prefix,turn):
        if len(prefix['actions_m'])!=turn+2 or len(prefix['observed_gaps_m'])!=turn+2:
            raise ValueError('Actual zero/weak prefix plus one observation per completed turn required')
        if self.controller is not None:
            return self.controller.plan(prefix['actions_m'],prefix['observed_gaps_m'],'continuous_guard',turn)
        if self.policy=='rest_repeat':proposal=0.
        elif self.policy=='weak_repeat':proposal=self.guard['weak_drive_m']
        else:proposal=self.settings['fixed_proposals_m'][turn]
        return proposal,{'policy':self.policy,'proposal_m':proposal,'round':turn,'uses_private_state':False}
