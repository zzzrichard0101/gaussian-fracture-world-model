from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from wmfracture.action_data import assert_group_split, project_to_bank, validate_inputs
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.cohesive_inference import known_action_rollout, reachable_state


@pytest.mark.parametrize("dt", [2.5e-6, 5e-6])
def test_vectorized_physics_matches_independent_fixture_and_preserves_history(dt):
    config = json.loads((Path(__file__).parents[1]/"config/cohesive-probe.json").read_text())
    states, reference = [], []
    times = [0., .004, .012, .036, .04, .08, .18]
    for peak, probe, impact in [(0., 0., .12), (.00065, .04, .245), (.00185, .04, .42)]:
        system = reachable_state(config, peak)
        states.append(type(system.state)(**asdict(system.state)))
        reference.append(known_action_rollout(system, 0., .18, dt,
            [{"id": "p", "time": .004, "left": probe, "right": 0.},
             {"id": "f", "time": .036, "left": impact, "right": 0.}], times))
    result = batch_rollout(config, states, [0., .04, .04], [.12, .245, .42], dt, times)
    for key in result:
        np.testing.assert_allclose(result[key], [[row[key] for row in rows] for rows in reference], atol=2e-10, rtol=0)
    assert np.all(np.diff(result["maximum_opening"], axis=1) >= 0)
    assert [state.maximum_opening for state in states] == [0., .00065, .00185]


def test_history_groups_cannot_leak_between_training_and_test():
    assert_group_split({"train": ["a", "b"], "validation": ["c"], "test": ["d"]})
    with pytest.raises(ValueError, match="leakage"):
        assert_group_split({"train": ["a"], "test": ["a"]})


def test_future_response_and_damage_cannot_be_prediction_input():
    arrays = {"rgbd_features": np.zeros((2, 45)), "prefix_index": np.arange(2), "actions": np.zeros((2, 2)),
        "accepted": np.ones((2, 13)), "rgbd_best": np.zeros(2), "bank_index": np.zeros(2)}
    validate_inputs(arrays)
    for key in ("future_gaps", "true_damage", "true_history"):
        with pytest.raises(ValueError, match="privileged"):
            validate_inputs(dict(arrays, **{key: np.zeros(2)}))


def test_projection_respects_observation_match_and_empty_set():
    bank = np.array([[0., 0.], [1., 1.], [2., 2.]])
    assert project_to_bank([0., 0.], [False, True, True], bank) == 1
    assert project_to_bank([0., 0.], [False, False, False], bank) == -1


def test_test_embeddings_cannot_change_fitted_pca_or_training_normalizer():
    from wmfracture.action_adapter import fit_pca, training_normalizer
    rng = np.random.default_rng(9)
    embeddings = rng.normal(size=(8, 12)).astype(np.float32)
    center, components = fit_pca(embeddings, [0, 1, 2, 3], 3)
    changed = embeddings.copy(); changed[4:] += 1000
    other_center, other_components = fit_pca(changed, [0, 1, 2, 3], 3)
    np.testing.assert_array_equal(center, other_center)
    np.testing.assert_array_equal(components, other_components)
    mean, std = training_normalizer(embeddings[:4])
    other_mean, other_std = training_normalizer(changed[:4])
    np.testing.assert_array_equal(mean, other_mean)
    np.testing.assert_array_equal(std, other_std)


def test_action_ablation_removes_only_future_action_input():
    from wmfracture.action_adapter import model_inputs
    arrays = {"actions": np.array([[.04, .12], [.04, .36]], dtype=np.float32),
        "rgbd_features": np.zeros((2, 45), dtype=np.float32), "prefix_index": np.array([0, 0])}
    embeddings = np.ones((1, 4), dtype=np.float32)
    center, components = np.zeros(4), np.eye(4)[:2]
    actual = model_inputs("rgbd_vjepa_action", arrays, embeddings, center, components)
    ablated = model_inputs("rgbd_vjepa_no_future_action", arrays, embeddings, center, components)
    assert not np.array_equal(actual[0], actual[1])
    np.testing.assert_array_equal(ablated[0], ablated[1])


def test_future_impulse_cannot_change_position_observed_before_that_impulse():
    import torch
    from wmfracture.action_adapter import DynamicsHead
    torch.manual_seed(3)
    model = DynamicsHead(5, 4, 16, future_action_column=4).eval()
    inputs = torch.zeros((2, 5))
    inputs[1, 4] = 2.
    outputs = model(inputs)
    assert outputs[0, 0].item() == outputs[1, 0].item()
    assert not torch.equal(outputs[0, 1:], outputs[1, 1:])
