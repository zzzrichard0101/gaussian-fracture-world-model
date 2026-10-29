from fractions import Fraction as Q

import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import equilibrium_cycle, damage
from wmfracture.continuous_risk import ContinuousRisk
from wmfracture.probe_identifiability import PairProbeDiagnostic, gaussian_pair_score


def diagnostic(cap=.02):
    return PairProbeDiagnostic(ContinuousRisk({
        'parameter_bounds': [[1.2, 3.], [.0009, .0023]], 'spring': 10000.,
        'law': {'area': .01, 'tensile_stiffness': 1e7, 'peak_traction': 1000.},
        'actions_m': [0., .001], 'observed_gaps_m': [0., 0.], 'reference_count': 2,
        'noise_radius_m': .00006, 'bias_bound_m': .00015, 'damage_cap': cap}))


def pair(d):
    return [(Q(13, 10), d.matched_preparation(Q(13, 10), Q(1, 2))),
            (Q(14, 5), d.matched_preparation(Q(14, 5), Q(1, 2)))]


def test_matched_states_have_identical_all_nondamaging_signals_and_same_stiffness():
    d = diagnostic(); a, b = pair(d)
    assert a[1] != b[1]
    assert d.compliance(*a) == d.compliance(*b) == Q(1, 2)
    assert d.model.damage(*a) == d.model.damage(*b)
    actions = [min(a[1], b[1])*Q(i, 64) for i in range(65)]
    np.testing.assert_array_equal(d.sequence_difference(a, b, actions), 0.)
    zero = diagnostic(0.); result = zero.maximum_difference(a, b, .00185)
    assert result['best']['difference'] == 0


@pytest.mark.parametrize('q', [Q(1, 5), Q(1, 3), Q(1, 2), Q(5, 8)])
def test_exact_branch_maximum_matches_dense_scan_and_independent_cycle_solver(q):
    d = diagnostic()
    a, b = [(e, d.matched_preparation(e, q)) for e in (Q(13, 10), Q(14, 5))]
    result = d.maximum_difference(a, b, .00185)
    amplitude = float(result['ceiling'])
    # Dense scan is a check, not the source of the maximum bound.
    actions = np.linspace(0, amplitude, 10001)
    law, fixture = InterfaceLaw(area=.01), Fixture()
    gaps = []
    for e, h in (a, b):
        _, maximum = equilibrium_cycle(0., float(e), float(h), law, fixture)
        before = float(damage(maximum, float(e), law))
        response, after = equilibrium_cycle(maximum, float(e), actions, law, fixture)
        gaps.append(response)
        assert np.max(damage(after, float(e), law)-before) <= .02+1e-13
    dense = float(np.max(abs(gaps[0]-gaps[1])))
    assert dense <= float(result['best']['difference'])+1e-16
    assert dense == pytest.approx(float(result['best']['difference']), abs=5e-8)
    e, h = a
    beyond = result['ceiling']+Q(1, 10**8)
    if result['ceiling'] < Q(185, 100000):
        assert any(d.model.point_risk(e, h, beyond)[0] > d.model.cap for e, h in (a, b))


def test_prefix_budget_is_not_reset_and_unsafe_sequence_is_rejected():
    d = diagnostic(); a, b = pair(d); result = d.maximum_difference(a, b, .00185)
    actions = [result['best']['action'], 0., result['best']['action']]
    delta = d.sequence_difference(a, b, actions)
    assert delta[0] == delta[2]
    rng = np.random.default_rng(101)
    for _ in range(10):
        actions = rng.uniform(0., float(result['ceiling'])-1e-12, 40)
        assert np.max(abs(d.sequence_difference(a, b, actions))) <= float(result['best']['difference'])
    with pytest.raises(ValueError): d.sequence_difference(a, b, [.003])


def test_gaussian_score_matches_dense_covariance_and_repetition_is_informative():
    delta = np.array([0., 1e-5, 1e-5, 0.])
    sigma, tau = 2e-5, 5e-5
    expected = delta@np.linalg.solve(sigma*sigma*np.eye(4)+tau*tau*np.ones((4, 4)), delta)
    assert gaussian_pair_score(delta, sigma, tau)['mahalanobis_squared'] == pytest.approx(expected)
    one = gaussian_pair_score([1e-5], sigma)
    repeated = gaussian_pair_score([1e-5]*32, sigma)
    assert repeated['equal_prior_bayes_error'] < one['equal_prior_bayes_error'] < .5
    assert repeated['kl_nats'] == pytest.approx(32*one['kl_nats'])
    assert gaussian_pair_score([0., 0.], sigma, tau)['equal_prior_bayes_error'] == .5


def test_invalid_parameters_and_open_separation_endpoint_are_rejected():
    d = diagnostic()
    with pytest.raises(ValueError): d.matched_preparation(1.3, 1.)
    with pytest.raises(ValueError): d.state((.5, .001))
    with pytest.raises(ValueError): d.amplitude_ceiling(pair(d)[0], -1.)
    with pytest.raises(ValueError): diagnostic(.5).amplitude_ceiling(pair(d)[0], .002)
    with pytest.raises(ValueError): gaussian_pair_score([1.], 0.)
