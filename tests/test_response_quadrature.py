import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import PrefixModel
from wmfracture.measurement_bias import GaussianObservation
from wmfracture.response_quadrature import ResponseQuadrature,log_normal_mass


def engine(actions=None,observed=None,sigma=.000001):
    return ResponseQuadrature(InterfaceLaw(area=.01),Fixture(),[[1.2,3.],[.0009,.0023]],
        [0.,.001] if actions is None else actions,[.00003,.00063] if observed is None else observed,sigma,.00005)


def test_remote_response_matches_independent_existing_cycle_solver():
    q=engine();rng=np.random.default_rng(71);unit=rng.random((200,2))
    old=PrefixModel(q.law,q.fixture,q.bounds);parameters=old.parameters(unit)
    actions=[0.,.001,.0013,.0007,.00185,0.,.0027,.001]
    prediction,_=old.predict(unit,actions)
    np.testing.assert_allclose(q.predict(parameters,actions),prediction,atol=2e-17,rtol=2e-12)
    np.testing.assert_allclose(q.gaussian_score(q.predict(parameters,q.actions)),
        GaussianObservation('shared',q.tau)(q.predict(parameters,q.actions),q.observed,q.sigma),atol=1e-9,rtol=1e-12)


def test_prior_mass_and_moments_preserve_elastic_interval_and_jacobian():
    q=engine(actions=[0.],observed=[0.]);result=q.integrate([.003],energy_panels=8,inner_order=20)
    e,h=result.parameters.T;w=result.weights
    assert np.exp(result.log_integral)==pytest.approx(1.,abs=1e-12)
    assert w.sum()==pytest.approx(1.,abs=1e-13)
    assert w@e==pytest.approx(2.1,abs=1e-12)
    assert w@h==pytest.approx(.0016,abs=1e-14)
    assert w@(h*h)==pytest.approx((.0009**2+.0009*.0023+.0023**2)/3,abs=1e-16)
    assert w@(h<=q.onset)==pytest.approx((q.onset-.0009)/(.0023-.0009),abs=1e-12)


@pytest.mark.parametrize('sigma',[1e-6,2e-5])
def test_conditional_integral_matches_direct_history_quadrature(sigma):
    q=engine(sigma=sigma);energies=np.array([1.25,2.,2.9])
    h,lw=q.conditional_nodes(energies,64)
    for index,e in enumerate(energies):
        # Independent integration in physical H; provide a dense initial
        # partition so the narrow response ridge cannot be skipped.
        scale=float(np.max(lw[index]))
        def density(history):
            return float(np.exp(q.gaussian_score(q.predict(np.array([[e,history]]),q.actions))[0]-scale))
        direct,error=quad(density,.0009,.0023,points=np.linspace(.0009,.0023,201)[1:-1],epsabs=1e-10,epsrel=1e-9,limit=600)
        transformed=float(np.exp(logsumexp(lw[index])-scale))
        assert transformed==pytest.approx(direct,rel=2e-5,abs=1e-9)


def test_tail_mass_is_finite_for_extreme_normal_intervals():
    values=log_normal_mass(np.array([-1001.,10.,-.1]),np.array([-1000.,11.,.1]))
    assert np.isfinite(values).all() and (values<0).all()


def test_refinement_agrees_for_narrow_repeated_observations():
    q=engine(actions=[0.,.001,.001,.001,.001],observed=[.00003,.00063,.0006303,.0006297,.0006301])
    results=[q.forecast(q.integrate([.0026,.0031,.0035],energy_panels=p,inner_order=p),[.0026,.0031,.0035]) for p in (16,32)]
    np.testing.assert_allclose([r['mean_gap_m'] for r in results[0]],[r['mean_gap_m'] for r in results[1]],atol=1e-7,rtol=1e-4)


def test_prepared_separation_and_invalid_public_inputs_rejected():
    with pytest.raises(ValueError):ResponseQuadrature(InterfaceLaw(area=.01),Fixture(),[[1.2,3.],[.0009,.003]],[0.],[0.],1e-6,5e-5)
    with pytest.raises(ValueError):engine(actions=[0.,-1.],observed=[0.,0.])


def test_broad_nonzero_likelihood_preserves_uniform_physical_measure():
    q=engine(actions=[0.,.001],observed=[0.,0.],sigma=1.)
    result=q.integrate([.0031],energy_panels=8,inner_order=32)
    assert np.exp(result.log_integral)==pytest.approx(1.,abs=1e-6)
    assert result.weights@result.parameters[:,1]==pytest.approx(.0016,abs=3e-9)


def test_separation_mass_handles_energy_event_boundary_exactly_in_prior():
    q=engine(actions=[0.],observed=[0.]);r=q.integrate([.0031],energy_panels=8,inner_order=8)
    forecast=q.forecast(r,[.0031])[0]
    assert forecast['separation_mass']==pytest.approx((1.55-1.2)/(3.-1.2),abs=1e-12)


def test_elastic_prior_interval_keeps_nonzero_posterior_mass():
    q=engine(actions=[0.,.001],observed=[0.,.001/11])
    result=q.integrate([.0031],energy_panels=16,inner_order=32)
    assert result.weights@(result.parameters[:,1]<q.onset)>.9


def test_independent_outer_reference_agrees_with_refined_composite_rule():
    q=engine();r=q.integrate([.0026,.0031,.0035],energy_panels=32,inner_order=32)
    expected=q.forecast(r,[.0026,.0031,.0035])
    reference=q.adaptive_reference([.0026,.0031,.0035],r.log_integral,inner_order=48,panels=16)
    assert reference['success']
    np.testing.assert_allclose([f['mean_gap_m'] for f in expected],[f['mean_gap_m'] for f in reference['futures']],atol=2e-8)
    np.testing.assert_allclose([f['separation_mass'] for f in expected],[f['separation_mass'] for f in reference['futures']],atol=2e-5)
