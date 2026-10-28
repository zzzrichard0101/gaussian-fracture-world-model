from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from wmfracture.belief_state import energy_weights, mixture_diagnostics, state_targets, validate_evidence
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.cohesive_inference import known_action_rollout, reachable_state


def test_future_plans_and_response_targets_cannot_enter_belief_inputs():
    arrays = {"prefix_index": np.array([0]), "scores": np.zeros((1, 13)), "accepted": np.ones((1, 13)),
        "rgbd_features": np.zeros((1, 45)), "probe": np.zeros((1, 1))}
    validate_evidence(arrays)
    for key in ("future_action", "future_response", "bank_index", "true_history"):
        with pytest.raises(ValueError, match="exclude future"):
            validate_evidence(dict(arrays, **{key: np.zeros(1)}))


def test_equal_observations_preserve_prior_and_empty_matches_remain_empty():
    prior = np.array([.2, .3, .5])
    weights = energy_weights(np.ones((2, 3)), np.array([[1, 1, 1], [0, 0, 0]]), prior, .001)
    np.testing.assert_allclose(weights[0], prior, atol=1e-15)
    np.testing.assert_array_equal(weights[1], np.zeros(3))


def test_barycentric_targets_preserve_history_coordinate_without_averaging_states():
    candidates = [0., .001, .002, .003]
    targets = state_targets([.00065, .00185, .003], candidates)
    np.testing.assert_allclose(targets.sum(1), 1.)
    np.testing.assert_allclose(targets@candidates, [.00065, .00185, .003])


def test_mixture_scores_do_not_pretend_mean_is_a_real_candidate():
    bank = np.array([[0., 0.], [2., 2.]])
    score = mixture_diagnostics([.5, .5], bank, [0., 1.], np.array([1., 1.]), False)
    assert score["mean_gap_mse_m2"] == 0.
    assert score["map_gap_mse_m2"] == 1.
    assert score["path_energy_m"] == pytest.approx(.5)
    assert score["marginal_crps_m"] == pytest.approx(.5)
    assert score["failure_probability"] == .5
    assert score["effective_candidates"] == 2.


def test_restarted_candidate_matches_uninterrupted_known_impulse_rollout():
    config = json.loads((Path(__file__).parents[1]/"config/cohesive-probe.json").read_text())
    system = reachable_state(config, .00185)
    known_action_rollout(system, 0., .012, 5e-6,
        [{"id": "probe", "time": .004, "left": .04, "right": 0.}], [.012])
    before = asdict(system.state)
    times = [.036, .052, .06, .18]
    resumed = batch_rollout(config, [system.state], [0.], [.325], 5e-6, times,
        future_time=.052, start_elapsed=.012)
    assert asdict(system.state) == before
    reference = known_action_rollout(system, .012, .18, 5e-6,
        [{"id": "delayed-impact", "time": .052, "left": .325, "right": 0.}], times)
    for key in resumed:
        np.testing.assert_allclose(resumed[key][0], [r[key] for r in reference], atol=2e-10, rtol=0)


def test_zero_evidence_gate_disables_learned_state_correction():
    import torch
    from wmfracture.belief_network import BeliefCorrection
    model = BeliefCorrection(4, 8)
    with torch.no_grad():
        model.network[-1].bias.fill_(3.)
    result = model(torch.randn(2, 4), torch.zeros(2))
    assert torch.equal(result, torch.zeros_like(result))
    active = model(torch.randn(2, 4), torch.ones(2))
    assert torch.max(active.abs()) <= 1.
