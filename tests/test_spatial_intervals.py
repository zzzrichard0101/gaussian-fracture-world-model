from fractions import Fraction as Q

import numpy as np
import pytest

from wmfracture.spatial_intervals import Interval, SpatialEnclosure, damage_bounds, damage_difference, bias_intervals
from wmfracture.spatial_observation import BatchSpatial


def contains(interval, value):
    return np.all(np.asarray(value) >= interval.lo) and np.all(np.asarray(value) <= interval.hi)


def test_outward_primitives_enclose_exact_binary_float_rationals():
    rng = np.random.default_rng(2026103031)
    for a, b in rng.uniform(-4., 4., (100, 2)):
        qa, qb = Q.from_float(float(a)), Q.from_float(float(b))
        for result, exact in ((Interval(a)+b, qa+qb), (Interval(a)-b, qa-qb),
                              (Interval(a)*b, qa*qb), (Interval(a)/b, qa/qb)):
            assert Q.from_float(float(result.lo)) <= exact <= Q.from_float(float(result.hi))
    with pytest.raises(ArithmeticError): Interval(1.)/Interval(-1., 1.)


def test_damage_and_coupled_difference_include_mode_switches_and_energy_peak():
    energies = np.linspace(1.3, 2.8, 31)
    for before, after in ((0., .00005), (.00005, .003), (.0007, .0012), (.0027, .0031), (.006, .007)):
        bound = damage_difference(Interval(1.3, 2.8), Interval(before), Interval(after))
        for e in energies:
            physics = BatchSpatial([[e, 0., 0.]])
            first = physics.damage(np.full((1, 3), before))[0, 0]
            second = physics.damage(np.full((1, 3), after))[0, 0]
            assert float(bound.lo)-1e-14 <= second-first <= float(bound.hi)+1e-14
    zero = damage_difference(Interval(1.3, 2.8), Interval(.001, .002), Interval(.001, .001))
    assert zero.hi == 0.


def test_endpoint_mode_union_encloses_random_interior_states_and_contact():
    enclosure = SpatialEnclosure([[1.3, 2.8], [.0008, .0018], [.0008, .0018]], 4)
    h = Interval([0., .0004, .0001], [.001, .0014, .0009])
    drive = Interval([.0002, 0.], [.0016, .0009])
    pose, maximum = enclosure.endpoint(drive, h)
    rng = np.random.default_rng(39)
    for _ in range(30):
        e = rng.uniform(1.3, 2.8)
        own_h = rng.uniform(h.lo, h.hi); own_drive = rng.uniform(drive.lo, drive.hi)
        q, actual = BatchSpatial([[e, 0., 0.]]).endpoint(own_drive, own_h[None, :])
        assert contains(pose, q[0])
        assert contains(maximum, actual[0])


def test_point_history_replay_covers_existing_batch_solver():
    parameters = np.array([1.3803881930187345, .0014446525566279888, .001244484293460846])
    box = np.column_stack((parameters, parameters))
    enclosure = SpatialEnclosure(box, 8)
    actions = [[0., 0.], [.0008, .0008]]
    result = enclosure.evaluate(actions, [-.04, .04], [.0004, .00145])
    physics = BatchSpatial(parameters[None, :], 8)
    _, reference = physics.cycle(np.zeros((1, 3)), parameters[1:])
    history = reference.copy(); predicted = []
    for drive in actions:
        q, history = physics.cycle(history, drive)
        predicted.append([q[0, 0]-q[0, 1], q[0, 0]+q[0, 1]])
    _, final = physics.cycle(history, [.0004, .00145])
    assert contains(result['prediction'], predicted)
    assert contains(result['reference'], reference[0])
    assert contains(result['after'], final[0])
    risk = np.max(physics.damage(final)-physics.damage(reference))
    assert contains(result['risk'], risk)


def test_outer_and_inner_bias_intervals_have_different_quantifiers():
    prediction = Interval([[.0, .0], [.0004, .0005]], [[.000002, .000002], [.000402, .000502]])
    observed = np.array([[.00003, -.00003], [.00043, .00047]])
    outer = bias_intervals(prediction, observed, .00001, .00015)
    inner = bias_intervals(prediction, observed, .00001, .00015, inner=True)
    assert np.all(outer[0] <= inner[0]) and np.all(inner[1] <= outer[1])
    assert np.all(inner[0] <= inner[1])
    bias = (inner[0]+inner[1])/2
    assert np.all(abs(observed-prediction.lo-bias) <= .00001)
    assert np.all(abs(observed-prediction.hi-bias) <= .00001)


def test_zero_drive_preserves_uncertain_history_exactly():
    enclosure = SpatialEnclosure([[1.3, 2.8], [.0008, .0018], [.0008, .0018]], 4)
    history = Interval([.001, .002, .003], [.0011, .0021, .0031])
    q, after = enclosure.cycle(history, Interval([0., 0.]))
    np.testing.assert_array_equal(q.lo, [0., 0.]); np.testing.assert_array_equal(q.hi, [0., 0.])
    np.testing.assert_array_equal(after.lo, history.lo); np.testing.assert_array_equal(after.hi, history.hi)


def test_invalid_boxes_and_unstable_domain_rejected():
    with pytest.raises(ValueError): SpatialEnclosure([[.5, 2.], [0., .001], [0., .001]])
    with pytest.raises(ValueError): SpatialEnclosure([[2., 1.], [0., .001], [0., .001]])
    with pytest.raises(ArithmeticError): Interval(np.nan)
