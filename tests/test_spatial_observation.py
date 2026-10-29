import inspect

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from wmfracture.spatial_fixture import three_zone_fixture
from wmfracture.spatial_observation import (BatchSpatial, PhysicalCatalog, bounded_bias_posterior,
    infer_public, tilt_proposal, choose_public)


def settings():
    return {'zero_action': 0, 'initial_weak_action': 1, 'left_weak_action': 2, 'right_weak_action': 3,
        'left_mid_action': 4, 'right_mid_action': 5, 'tilt_threshold_sigma': 2.,
        'bias_sigma_m': .00005, 'bias_bound_m': .00015, 'noise_multiple': 3., 'damage_cap': .02}


def catalog():
    return PhysicalCatalog([[1.3, .0018, .0008], [2.8, .0008, .0018], [2., .0012, .0012]],
        [[0., 0.], [.0008, .0008], [.001, .0004], [.0004, .001], [.00145, .0004], [.0004, .00145]],
        [[.0031, .0031]], increments=64)


def test_vectorized_endpoint_matches_independent_mode_enumeration_including_contact_and_failure():
    parameters = np.array([[1.3, .00183, .00081], [2., .0012, .0012], [2.8, .00081, .00183]])
    batch = BatchSpatial(parameters); maximum = np.zeros((3, 3))
    rng = np.random.default_rng(3021)
    for drive in list(rng.uniform(-.0002, .0045, (15, 2)))+[[0., 0.], [-.0001, -.0001]]:
        q, updated = batch.endpoint(drive, maximum)
        for i, energy in enumerate(parameters[:, 0]):
            reference = three_zone_fixture(float(energy))
            own, history, _ = reference.equilibrium(drive, maximum[i])
            np.testing.assert_allclose(q[i], own, atol=2e-14)
            np.testing.assert_allclose(updated[i], history, atol=2e-14)
        maximum = updated


def test_preparation_and_complete_sequential_path_match_small_increment_reference():
    parameters = [[1.3, .00183, .00081], [2.8, .00081, .00183]]
    batch = BatchSpatial(parameters, 128); maximum = np.zeros((2, 3))
    references = [three_zone_fixture(p[0]) for p in parameters]
    states = [r.virgin() for r in references]
    for commands in (np.array(parameters)[:, 1:], np.array([[.00145, .0004], [.0004, .00145]]),
                     np.array([[.0032, .0006], [.0006, .0032]])):
        q, maximum = batch.cycle(maximum, commands)
        for i, (ref, state) in enumerate(zip(references, states)):
            rows, _ = ref.segment(state, commands[i], 2e-6)
            np.testing.assert_allclose(q[i], state.q, atol=2e-7, rtol=0.)
            ref.segment(state, [0., 0.], 2e-6)
            np.testing.assert_allclose(maximum[i], state.maximum_opening, atol=2e-7, rtol=0.)


def test_truncated_shared_bias_likelihood_matches_direct_bias_integration():
    p = np.array([[[0.], [.0004], [.0005]], [[0.], [.00042], [.00049]]])
    y = np.array([[.00003], [.00044], [.00052]])
    sigma, tau, bound = .00002, .00005, .00015
    result = bounded_bias_posterior(p, y, sigma, tau)
    logz = []
    for prediction in p:
        residual = (y-prediction).ravel()
        lo = max(-bound, float(np.max(residual-3*sigma)))
        hi = min(bound, float(np.min(residual+3*sigma)))
        # Integrate in bias/sigma units, omitting hypothesis-independent factors.
        value, _ = quad(lambda t: np.exp(-.5*((t*sigma/tau)**2+np.sum((residual/sigma-t)**2))),
                        lo/sigma, hi/sigma, epsabs=1e-13)
        logz.append(np.log(value))
    expected = np.exp(logz-logsumexp(logz))
    np.testing.assert_allclose(result['weights'], expected, rtol=1e-11)


def test_bias_persists_across_frames_and_empty_catalog_abstains():
    p = np.array([[[0.], [.001]], [[0.], [.0005]]])
    first = bounded_bias_posterior(p[:, :1], [[.00003]], 1e-6, 5e-5)
    assert first['feasible'].all()
    second = bounded_bias_posterior(p, [[.00003], [.00103]], 1e-6, 5e-5)
    np.testing.assert_array_equal(second['feasible'], [True, False])
    assert second['weights'][0] == pytest.approx(1.)
    empty = bounded_bias_posterior(p, [[1.], [1.]], 1e-6, 5e-5)
    assert empty['weights'] is None and not empty['feasible'].any()


def test_zero_mass_closed_error_boundary_is_retained_for_finite_risk_checks():
    sigma, bound = 1/1024, 1/8
    result = bounded_bias_posterior(np.zeros((1, 2, 1)), [[bound+3*sigma], [bound-3*sigma]],
        sigma, 1/16, bias_bound=bound)
    assert result['feasible'][0]
    assert result['weights'] is None
    assert result['bias_lower_m'][0, 0] == result['bias_upper_m'][0, 0] == bound


def test_observation_positions_change_available_information_at_equal_readout_count():
    c = catalog(); path = (0, 1)
    center = c.observations(path, [0., 0.]); ends = c.observations(path, [-.04, .04])
    np.testing.assert_array_equal(center[:, 0], center[:, 1])
    assert np.max(abs(ends[:, 1]-ends[:, 0])) > 1e-4
    assert center.shape == ends.shape == (3, 2)


def test_tilt_rule_uses_bias_corrected_observations_and_center_falls_back():
    s = settings(); observed = [[.00003, -.00003], [.00043, .00007]]
    assert tilt_proposal([0, 1], observed, [-.04, .04], 1e-6, 0, s) == 4
    assert tilt_proposal([0, 1], observed, [0., 0.], 1e-6, 1, s) == 5
    shifted = np.asarray(observed)+[.00007, -.00008]
    assert tilt_proposal([0, 1], shifted, [-.04, .04], 1e-6, 0, s) == 4


def test_public_controller_signature_and_finite_guard_check_all_feasible_hypotheses():
    c = catalog(); s = settings(); path = [0, 1]
    y = [c.observations(path[:i+1], [-.04, .04])[0].tolist() for i in range(2)]
    action, posterior, log = choose_public('finite_guard_tilt', c, path, y, [-.04, .04], 2e-5, 0, s)
    assert action is not None
    selected = tuple(path)+(action,)
    assert np.max(c.damage_increment(selected)[posterior['feasible']]) <= .02+1e-12
    assert not c.separated(selected)[posterior['feasible']].any()
    assert list(inspect.signature(choose_public).parameters) == ['policy', 'catalog', 'actions', 'observed', 'positions', 'sigma', 'turn', 'settings']
    repeat, _, repeat_log = choose_public('finite_guard_tilt', c, path, y, [-.04, .04], 2e-5, 0, s)
    assert repeat == action and repeat_log == log


def test_public_posterior_replays_actions_and_never_initializes_new_hidden_history():
    c = catalog(); positions = [-.04, .04]; path = [0, 1, 4]
    history = c.transition(tuple(path)).copy()
    y = [c.observations(path[:i+1], positions)[0] for i in range(len(path))]
    result = infer_public(c, path, y, positions, 1e-6, settings())
    assert result['feasible'][0] and result['weights'][0] > 0.
    np.testing.assert_array_equal(c.transition(tuple(path)), history)
    future, events = c.future(tuple(path))
    assert future.shape == events.shape == (3, 1, 3)
    np.testing.assert_array_equal(c.transition(tuple(path)), history)


def test_invalid_batch_parameters_and_prefixes_rejected():
    with pytest.raises(ValueError): BatchSpatial([[.5, 0., 0.]])
    with pytest.raises(ValueError): BatchSpatial([[2., 0., 0.]], 0)
    c = catalog()
    with pytest.raises(ValueError): c.transition((99,))
