"""Response-coordinate quadrature for the existing scalar Gaussian target.

This changes integration coordinates, not the physical prior, likelihood or
risk certificate. Uniform E/H measure includes elastic intervals and the exact
H-to-compliance Jacobian. No prepared separation is supported in this version.
"""
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_ndtr, logsumexp, ndtr
from scipy.stats import truncnorm


def log_normal_mass(lower,upper):
    lower,upper=np.broadcast_arrays(np.asarray(lower,float),np.asarray(upper,float))
    if np.any(lower>=upper):raise ValueError('Ordered standardized interval required')
    result=np.empty(lower.shape)
    positive=lower>=0;negative=upper<=0;crossing=~(positive|negative)
    for mask,a,b in ((positive,-lower,-upper),(negative,upper,lower)):
        first,second=log_ndtr(a[mask]),log_ndtr(b[mask])
        result[mask]=first+np.log(-np.expm1(second-first))
    result[crossing]=np.log(ndtr(upper[crossing])-ndtr(lower[crossing]))
    return result


@dataclass
class QuadratureResult:
    parameters: np.ndarray
    weights: np.ndarray
    log_integral: float
    diagnostics: dict


class ResponseQuadrature:
    """Public histories only; same uniform physical prior/shared Gaussian proxy."""
    def __init__(self,law,fixture,bounds,actions,observed,sigma,bias_sigma):
        self.law,self.fixture=law,fixture
        self.bounds=np.asarray(bounds,float);self.actions=np.asarray(actions,float)
        self.observed=np.asarray(observed,float);self.sigma=float(sigma);self.tau=float(bias_sigma)
        if (self.bounds.shape!=(2,2) or not np.isfinite(self.bounds).all() or (np.diff(self.bounds)<=0).any()
                or self.bounds[0,0]<=0 or self.bounds[1,0]<0 or self.actions.ndim!=1 or not len(self.actions)
                or self.observed.shape!=self.actions.shape or not np.isfinite(self.actions).all()
                or not np.isfinite(self.observed).all() or (self.actions<0).any()
                or not np.isfinite([sigma,bias_sigma]).all() or sigma<=0 or bias_sigma<0):
            raise ValueError('Finite ordered prior and complete nonnegative public history required')
        self.area,self.k,self.strength,self.spring=law.area,law.tensile_stiffness,law.peak_traction,fixture.spring
        self.d0=self.strength/self.k;self.c=self.area*self.strength/self.spring;self.onset=self.d0+self.c
        if min(self.area,self.k,self.strength,self.spring)<=0:raise ValueError('Positive known mechanics required')
        if 2*self.bounds[0,0]/self.strength<=max(self.onset,self.bounds[1,1]):
            raise ValueError('Stable branch and no prepared separation over the whole prior required')
        self.q0=self.spring/(self.spring+self.area*self.k)
        self.peaks=np.maximum.accumulate(self.actions)
        hlo,hhi=self.bounds[1]
        self.hcuts=np.unique(np.r_[hlo,hhi,np.clip(np.r_[self.onset,self.peaks],hlo,hhi)])

    def compliance(self,energy,remote):
        energy,remote=np.broadcast_arrays(np.asarray(energy,float),np.asarray(remote,float))
        final=2*energy/self.strength;alpha=(final-self.d0)/(final-self.onset)
        beta=self.c*final/(final-self.onset)
        return np.where(remote<=self.onset,self.q0,np.where(remote>=final,1.,alpha-beta/np.maximum(remote,self.onset)))

    def predict(self,parameters,actions):
        parameters=np.asarray(parameters,float);actions=np.asarray(actions,float)
        peaks=np.maximum.accumulate(actions)
        return actions[None,:]*self.compliance(parameters[:,0,None],np.maximum(parameters[:,1,None],peaks[None,:]))

    def gaussian_score(self,prediction):
        residual=self.observed-np.asarray(prediction)
        mean=residual.mean(axis=-1);centered=residual-mean[...,None];n=len(self.actions)
        return -.5*(np.sum((centered/self.sigma)**2,axis=-1)+n*mean**2/(self.sigma**2+n*self.tau**2))

    def conditional_nodes(self,energies,inner_order):
        """H quadrature, returning E x node arrays with UNnormalized log weights."""
        energy=np.asarray(energies,float)
        if energy.ndim!=1 or not len(energy) or not np.isfinite(energy).all() or (energy<self.bounds[0,0]).any() or (energy>self.bounds[0,1]).any():
            raise ValueError('Energy nodes must belong to the prior')
        if type(inner_order) is not int or inner_order<2:raise ValueError('Inner order >=2 required')
        base,weights=leggauss(inner_order);t=(base+1)/2;weights=weights/2
        final=2*energy/self.strength;alpha=(final-self.d0)/(final-self.onset);beta=self.c*final/(final-self.onset)
        histories=[];logs=[];n=len(self.actions)
        for lo,hi in zip(self.hcuts[:-1],self.hcuts[1:]):
            middle=(lo+hi)/2
            u=np.where(self.peaks<middle,self.actions,0.)
            v=np.where(self.peaks[None,:]>=middle,self.actions[None,:]*self.compliance(energy[:,None],self.peaks[None,:]),0.)
            uc=u-u.mean();curvature=np.sum((uc/self.sigma)**2)+n*u.mean()**2/(self.sigma**2+n*self.tau**2)
            if hi<=self.onset or curvature==0:
                # Uniform H mass: includes all elastic prior, never drop atoms
                # in the observable compliance distribution.
                h=np.broadcast_to(lo+(hi-lo)*t,(len(energy),inner_order))
                prediction=self.actions[None,None,:]*self.compliance(energy[:,None,None],np.maximum(h[:,:,None],self.peaks[None,None,:]))
                logw=self.gaussian_score(prediction)+np.log((hi-lo)*weights)[None,:]
            else:
                residual=self.observed[None,:]-v;rm=residual.mean(axis=1)
                linear=np.sum((residual-rm[:,None])*uc[None,:],axis=1)/self.sigma**2+n*rm*u.mean()/(self.sigma**2+n*self.tau**2)
                mean=linear/curvature;std=1/np.sqrt(curvature)
                qlo,qhi=alpha-beta/lo,alpha-beta/hi
                zlo,zhi=(qlo-mean)/std,(qhi-mean)/std
                z=truncnorm.ppf(t[None,:],zlo[:,None],zhi[:,None])
                q=mean[:,None]+std*z
                h=beta[:,None]/(alpha[:,None]-q)
                # Roundoff clipping does not substitute for quadrature accuracy.
                h=np.clip(h,lo,hi)
                jacobian=beta[:,None]/(alpha[:,None]-q)**2
                peak=self.gaussian_score(v+mean[:,None]*u[None,:])
                logw=(peak+np.log(np.sqrt(2*np.pi)*std)+log_normal_mass(zlo,zhi))[:,None]+np.log(jacobian*weights[None,:])
            histories.append(h);logs.append(logw)
        h=np.concatenate(histories,axis=1);logw=np.concatenate(logs,axis=1)
        if not np.isfinite(h).all() or not np.isfinite(logw).all():
            raise FloatingPointError('Nonfinite quadrature nodes/weights; no prediction may be claimed')
        return h,logw

    def integrate(self,future_actions,energy_panels=16,energy_order=8,inner_order=16):
        if type(energy_panels) is not int or energy_panels<1 or type(energy_order) is not int or energy_order<2:
            raise ValueError('Positive energy panels and energy order >=2 required')
        futures=np.asarray(future_actions,float)
        if futures.ndim!=1 or not np.isfinite(futures).all() or (futures<0).any():raise ValueError('Finite nonnegative future actions required')
        elo,ehi=self.bounds[0]
        # Include all known separation mode boundaries; a fixed uniform scan
        # alone is not trusted to handle discontinuous event probabilities.
        switches=np.r_[self.actions,futures,self.hcuts]*self.strength/2
        cuts=np.unique(np.r_[np.linspace(elo,ehi,energy_panels+1),np.clip(switches,elo,ehi)])
        nodes,weights=leggauss(energy_order)
        energy=np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in zip(cuts[:-1],cuts[1:])])
        ew=np.concatenate([(b-a)/2*weights for a,b in zip(cuts[:-1],cuts[1:])])
        h,logw=self.conditional_nodes(energy,inner_order)
        logw+=np.log(ew)[:,None]-np.log(np.prod(np.ptp(self.bounds,axis=1)))
        logz=float(logsumexp(logw));probability=np.exp(logw-logz).ravel()
        parameters=np.column_stack((np.repeat(energy,h.shape[1]),h.ravel()))
        return QuadratureResult(parameters,probability,logz,{'energy_nodes':len(energy),'physical_nodes':len(probability),
            'energy_panels':energy_panels,'energy_order':energy_order,'inner_order':inner_order,
            'history_segments':len(self.hcuts)-1,'uniform_physical_prior_preserved':True,
            'scope':'Floating quadrature for unchanged Gaussian proxy; refinement is empirical, not a certified numerical error bound.'})

    def forecast(self,result,future_actions):
        energy,preparation=result.parameters.T;weights=result.weights;answers=[]
        current=max(self.actions)
        if len(future_actions) and min(future_actions)>=self.bounds[1,1]:
            # In this study future commands exceed all possible preparation
            # peaks. Marginalizing H first is exact for these forecast outputs.
            count=result.diagnostics['energy_nodes'];width=len(weights)//count
            energy=energy.reshape(count,width)[:,0]
            weights=weights.reshape(count,width).sum(axis=1)
            preparation=np.zeros(count)
        for action in future_actions:
            remote=np.maximum(preparation,max(current,action))
            gap=action*self.compliance(energy,remote)
            separated=remote>=2*energy/self.strength
            answers.append({'drive_m':float(action),'mean_gap_m':float(weights@gap),
                'separation_mass':float(weights@separated)})
        return answers

    def adaptive_reference(self,future_actions,log_scale,inner_order=64,panels=32,rtol=1e-7):
        """Different OUTER integrator, sharing the conditional H formulation.

        quad_vec's reported floating error is a diagnostic, not a proof. An
        explicit initial partition and known event boundaries reduce missed
        narrow peaks; agreement still needs refinement and separate H tests.
        """
        from scipy.integrate import quad_vec
        actions=list(future_actions);volume=np.prod(np.ptp(self.bounds,axis=1))
        def integrand(energy):
            h,logw=self.conditional_nodes(np.asarray([energy]),inner_order)
            lm=float(logsumexp(logw));weights=np.exp(logw.ravel()-lm)
            values=[1.]
            for action in actions:
                remote=np.maximum(h.ravel(),max(max(self.actions),action))
                gap=action*self.compliance(energy,remote)
                values.extend((float(weights@gap),float(weights@(remote>=2*energy/self.strength))))
            return np.exp(lm-log_scale)/volume*np.asarray(values)
        elo,ehi=self.bounds[0]
        points=np.unique(np.r_[np.linspace(elo,ehi,panels+1),np.clip(np.r_[self.actions,actions]*self.strength/2,elo,ehi)])
        value,error,info=quad_vec(integrand,elo,ehi,epsabs=1e-10,epsrel=rtol,points=points[1:-1],limit=1024,full_output=True)
        if not np.isfinite(value).all() or value[0]<=0:raise FloatingPointError('Invalid adaptive reference')
        futures=[{'drive_m':float(a),'mean_gap_m':float(value[1+2*i]/value[0]),
            'separation_mass':float(value[2+2*i]/value[0])} for i,a in enumerate(actions)]
        return {'futures':futures,'relative_mass':float(value[0]),'log_integral':float(log_scale+np.log(value[0])),
            'reported_vector_error':float(error),'success':bool(info.success),'status':int(info.status),
            'function_evaluations':int(info.neval),'inner_order':inner_order,'initial_panels':panels,
            'scope':'Adaptive outer reference shares conditional H nodes; no certified quadrature error.'}
