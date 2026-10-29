import numpy as np
import pytest

from wmfracture.spatial_continuous import estimate


def test_constant_likelihood_preserves_equal_volume_node_weights_and_does_not_claim_bounds():
    prediction = np.zeros((4, 2, 2))
    gaps = np.arange(12, dtype=float).reshape(4, 1, 3)*.0001
    events = gaps > .0006
    result = estimate(prediction, gaps, events, np.zeros((2, 2)), .00002,
        {'bias_sigma_m': .00005, 'noise_multiple': 3., 'bias_bound_m': .00015})
    np.testing.assert_allclose(result['mean_gap_m'], gaps.mean(axis=0))
    assert result['ess'] == pytest.approx(4.) and result['risk_upper_bound'] is None


def test_zero_feasible_nodes_is_not_a_continuous_domain_inconsistency_claim():
    result = estimate(np.zeros((4, 2, 2)), np.zeros((4, 1, 3)), np.zeros((4, 1, 3), bool),
        np.ones((2, 2)), .00002, {'bias_sigma_m': .00005, 'noise_multiple': 3., 'bias_bound_m': .00015})
    assert not result['available']
    assert result['status'] == 'no_feasible_nodes_not_domain_inconsistent'
    assert result['risk_upper_bound'] is None
