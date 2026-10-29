"""Standard Gaussian-ridge-guided integration cells; physical measure unchanged.

Within each known material mode, observations are affine in response q and
t=1/(2E/S-U0). Place E boundaries at conditional Gaussian/CDF transitions and
the residual Gaussian in t. Only public observations determine these cells.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from wmfracture.response_quadrature import ResponseQuadrature,QuadratureResult


class GuidedResponseQuadrature(ResponseQuadrature):
    def precision_inner(self,a,b):
        a,b=np.asarray(a),np.asarray(b);n=len(self.actions)
        return float(np.sum((a-a.mean())*(b-b.mean()))/self.sigma**2+
            n*a.mean()*b.mean()/(self.sigma**2+n*self.tau**2))

    def energy_cuts(self,future_actions,energy_panels):
        elo,ehi=self.bounds[0]
        modes=np.unique(np.r_[elo,ehi,np.clip(np.r_[self.actions,future_actions]*self.strength/2,elo,ehi)])
        cuts=list(modes)+list(np.linspace(elo,ehi,energy_panels+1))
        standardized=np.asarray([-10.,-6.,-3.,-1.,0.,1.,3.,6.,10.])
        p=self.peaks;safe=np.maximum(p,self.onset)
        for left,right in zip(modes[:-1],modes[1:]):
            final=2*((left+right)/2)/self.strength
            soft=(p>self.onset)&(p<final)
            q0=np.where(p<=self.onset,self.q0,np.where(p>=final,1.,1-self.c/safe))
            q1=np.where(soft,self.onset*(1-self.c/safe)-self.d0,0.)
            def add_t(values):
                values=np.asarray(values);values=values[np.isfinite(values)&(values>0)]
                energies=self.strength/2*(self.onset+1/values)
                cuts.extend(energies[(energies>left)&(energies<right)].tolist())
            for lo,hi in zip(self.hcuts[:-1],self.hcuts[1:]):
                midpoint=(lo+hi)/2;active=p<midpoint
                u=np.where(active,self.actions,0.)
                v0=np.where(active,0.,self.actions*q0)
                v1=np.where(active,0.,self.actions*q1)
                if hi<=self.onset:
                    v0=v0+u*self.q0;u=np.zeros_like(u)
                residual=self.observed-v0
                curvature=self.precision_inner(u,u)
                if curvature>0:
                    m0=self.precision_inner(u,residual)/curvature
                    m1=-self.precision_inner(u,v1)/curvature
                    std=1/np.sqrt(curvature)
                    for bound in (lo,hi):
                        q_const=1-self.c/bound;q_recip=self.onset*q_const-self.d0
                        slope=q_recip-m1
                        if slope!=0:add_t((standardized*std-q_const+m0)/slope)
                    projected=v1+u*m1
                    projected_residual=residual-u*m0
                else:
                    projected,projected_residual=v1,residual
                t_curvature=self.precision_inner(projected,projected)
                if t_curvature>0:
                    t_mean=self.precision_inner(projected,projected_residual)/t_curvature
                    add_t(t_mean+standardized/np.sqrt(t_curvature))
        # Exact duplicate boundaries are removed. Tiny distinct intervals are
        # retained rather than silently discarding prior probability mass.
        return np.unique(cuts)

    def integrate(self,future_actions,energy_panels=16,energy_order=8,inner_order=32):
        if type(energy_panels) is not int or energy_panels<1 or type(energy_order) is not int or energy_order<2:
            raise ValueError('Valid guided integration resolution required')
        future=np.asarray(future_actions,float)
        if future.ndim!=1 or not np.isfinite(future).all() or (future<0).any():raise ValueError('Nonnegative future queries required')
        cuts=self.energy_cuts(future,energy_panels);base,weights=leggauss(energy_order)
        energy=np.concatenate([(a+b)/2+(b-a)/2*base for a,b in zip(cuts[:-1],cuts[1:])])
        ew=np.concatenate([(b-a)/2*weights for a,b in zip(cuts[:-1],cuts[1:])])
        h,logw=self.conditional_nodes(energy,inner_order)
        logw+=np.log(ew)[:,None]-np.log(np.prod(np.ptp(self.bounds,axis=1)))
        logz=float(logsumexp(logw));probabilities=np.exp(logw-logz).ravel()
        parameters=np.column_stack((np.repeat(energy,h.shape[1]),h.ravel()))
        return QuadratureResult(parameters,probabilities,logz,{'energy_nodes':len(energy),'physical_nodes':len(probabilities),
            'energy_panels':energy_panels,'energy_order':energy_order,'inner_order':inner_order,
            'guided_energy_intervals':len(cuts)-1,'history_segments':len(self.hcuts)-1,
            'uniform_physical_prior_preserved':True,
            'scope':'Public-response-guided floating quadrature; no certified integration error.'})
