import numpy as np
import pytest

from wmfracture.world_model_io import causal_indices, context_canvas, hybrid_choice, validate_rank_arrays


def config():
    return {"image_size": 32, "patch_size": 16, "observation_slots": 8, "context_slots": 4,
            "frames_per_slot": 2, "tubelet_size": 2}


def test_context_target_boundary_does_not_split_temporal_patch():
    context, target = causal_indices(config())
    assert len(context) == len(target) == 16
    assert set(context).isdisjoint(target)
    assert np.array_equal(np.concatenate([context, target]), np.arange(32))
    bad = config(); bad["frames_per_slot"] = 1
    with pytest.raises(ValueError, match="tubelet"):
        causal_indices(bad)


def test_unknown_future_fill_does_not_change_observed_tensor_frames():
    prefix = np.random.default_rng(0).random((4, 32, 32, 3)).astype(np.float32)
    first, second = context_canvas(prefix, config(), 0.), context_canvas(prefix, config(), 1.)
    np.testing.assert_array_equal(first[:8], second[:8])
    assert not np.array_equal(first[8:], second[8:])


def test_ranking_contract_rejects_actual_future_and_changed_context():
    prefix = np.zeros((4, 32, 32, 3), dtype=np.float32)
    arrays = {"context_rgb": prefix, "candidate_rgb": np.zeros((2, 8, 32, 32, 3), dtype=np.float32),
              "context_roi": np.ones((32, 32), dtype=np.float32), "prefix_metrics": np.zeros((2, 4))}
    validate_rank_arrays(arrays, config())
    with pytest.raises(ValueError, match="private"):
        validate_rank_arrays(dict(arrays, actual_future=np.ones((1,))), config())
    arrays["candidate_rgb"][1, 0] = 1.
    with pytest.raises(ValueError, match="context"):
        validate_rank_arrays(arrays, config())


def test_empty_physics_match_set_does_not_get_world_model_override():
    assert hybrid_choice([], [.1, .2]) is None
    assert hybrid_choice([1, 2], [.01, .4, .2]) == 2
