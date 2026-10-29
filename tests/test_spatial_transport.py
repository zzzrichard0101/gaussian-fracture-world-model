import numpy as np
import pytest
import json
from pathlib import Path
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, qmc

from wmfracture.spatial_observation import bounded_bias_posterior
from wmfracture.spatial_transport import DefensiveMixture, bounded_loglike, summarize_importance, whiten, fit_proposals


def test_mixture_density_includes_every_overlapping_chart_and_uniform_component():
    means = np.array([[.4,.5,.6],[.6,.5,.4]])
    cov = np.array([np.diag([.02,.03,.04]),np.diag([.04,.03,.02])])
    mixture = DefensiveMixture(means,cov,.2); points=np.array([[.5,.5,.5],[1.1,.5,.5]])
    expected = .4*multivariate_normal.pdf(points,means[0],cov[0])+.4*multivariate_normal.pdf(points,means[1],cov[1])
    expected[0] += .2
    np.testing.assert_allclose(np.exp(mixture.log_density(points)),expected,rtol=1e-13)


def test_density_correction_preserves_original_uniform_prior_and_plateau_region_mass():
    mixture = DefensiveMixture([[.78,.72,.68]], [np.diag([.05,.06,.04])**2], .2)
    unit = qmc.Sobol(4,scramble=True,seed=31).random_base2(14)
    points,logq=mixture.sample(unit); inside=np.all((points>=0)&(points<=1),axis=1)
    w=np.where(inside,np.exp(-logq),0.)
    assert np.mean(w)==pytest.approx(1.,abs=.005)
    np.testing.assert_allclose(np.sum(w[:,None]*points,axis=0)/w.sum(),.5,atol=.005)
    # An uninformative/plateau subset cannot be dropped merely because most
    # response-conditioned charts lie elsewhere in parameter space.
    assert np.sum(w*(points[:,1]<.2))/w.sum()==pytest.approx(.2,abs=.005)
    assert np.mean(points[:,1]<.2)<.1


def test_outside_cube_draws_are_not_silently_renormalized_as_truncated_gaussians():
    mixture=DefensiveMixture([[.98,.98,.98]],[np.eye(3)*.1**2],.2)
    points,logq=mixture.sample(qmc.Sobol(4,scramble=True,seed=33).random_base2(12))
    inside=np.all((points>=0)&(points<=1),axis=1)
    assert np.count_nonzero(~inside)>500
    loglike=np.where(inside,0.,-np.inf)
    result=summarize_importance(loglike,logq,np.zeros((len(points),1,3)),np.zeros((len(points),1,3),bool))
    assert result['nodes']==len(points) and result['positive_weight_nodes']==int(inside.sum())
    assert result['risk_upper_bound'] is None


def test_log_likelihood_matches_existing_bounded_shared_bias_target():
    rng=np.random.default_rng(37); prediction=rng.uniform(.0002,.0003,(20,4,2))
    observed=np.full((4,2),.00025); settings={'bias_sigma_m':.00005,'noise_multiple':3.,'bias_bound_m':.00015}
    loglike=bounded_loglike(prediction,observed,.00002,settings)
    original=bounded_bias_posterior(prediction,observed,.00002,.00005)
    np.testing.assert_allclose(np.exp(loglike-logsumexp(loglike)),original['weights'],atol=1e-14)


def test_whitening_matches_direct_persistent_bias_quadratic_form():
    values=np.array([[[.1,.2],[.3,.4],[.5,.6]]]); sigma,tau=.2,.3
    transformed=whiten(values,sigma,tau)
    covariance=sigma*sigma*np.eye(3)+tau*tau*np.ones((3,3))
    expected=sum(values[0,:,i]@np.linalg.solve(covariance,values[0,:,i]) for i in range(2))
    assert np.sum(transformed**2)==pytest.approx(expected)


def test_invalid_or_singular_proposals_rejected():
    with pytest.raises(ValueError): DefensiveMixture([[0.,0.,0.]],[np.eye(3)],0.)
    with pytest.raises(np.linalg.LinAlgError): DefensiveMixture([[0.,0.,0.]],[np.zeros((3,3))],.2)


def test_response_projection_handles_a_rank_two_map_without_a_global_inverse():
    class LinearForward:
        def evaluate(self, unit, prefix, future=False):
            values=np.zeros((len(unit),len(prefix),2))
            values[:,1,0]=.001*(unit[:,1]+.2*unit[:,0])
            values[:,1,1]=.001*(unit[:,2]-.3*unit[:,0])
            return values,None
    config=json.loads((Path(__file__).resolve().parents[1]/'config/spatial-transport.json').read_text())
    config['charts']=4
    model=LinearForward(); unit=qmc.Sobol(3,scramble=True,seed=44).random_base2(7)
    prediction,_=model.evaluate(unit,[0,1]); target,_=model.evaluate(np.array([[.4,.55,.45]]),[0,1])
    proposals,audit=fit_proposals(model,unit,prediction,[0,1],target[0],.000001,{'bias_sigma_m':.00005},config)
    projected=np.asarray(audit['projected_centers_unit']); fitted,_=model.evaluate(projected,[0,1])
    np.testing.assert_allclose(fitted[:,1,:],np.broadcast_to(target[0,1,:],(4,2)),atol=2e-8)
    np.testing.assert_array_equal(projected[:,0],np.asarray(audit['raw_centers_unit'])[:,0])
    assert audit['rank_counts']['2']==4
    assert np.linalg.eigvalsh(proposals['response_chart'].covariances).min()>0.
