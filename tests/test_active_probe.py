import numpy as np
import pytest

from wmfracture.active_probe import select_probe, task_weights, rgbd_summary


def test_guard_rejects_informative_but_damaging_probe_and_passive_is_counterfactual():
    actions = [0., .04, .20]
    features = np.zeros((2, 3, 2, 1)); features[1, 1] = 1.; features[1, 2] = 5.
    damage = np.array([[.2, .21, 1.], [.4, .41, .8]])
    generic = select_probe(actions, features, damage, "disagreement", [1.], .02)
    guarded = select_probe(actions, features, damage, "guarded", [1.], .02)
    assert generic["selected_impulse"] == .20 and guarded["selected_impulse"] == .04
    assert not guarded["eligible_actions"][2]
    np.testing.assert_allclose(guarded["max_candidate_added_damage"], [0., .01, .8])


def test_no_extra_information_or_one_candidate_returns_passive():
    for n in (1, 3):
        result = select_probe([0., .04, .20], np.zeros((n, 3, 2, 1)), np.full((n, 3), .9), "guarded", [1.], .02)
        assert result["selected_impulse"] == 0.


def test_new_separation_is_rejected_even_below_damage_cap():
    features = np.zeros((2, 2, 2, 1)); features[1, 1] = 8.
    result = select_probe([0., .20], features, np.array([[.999, 1.], [.99, 1.]]), "guarded", [1.], .02)
    assert result["candidate_new_separation"][1] and result["selected_impulse"] == 0.


def test_task_weights_use_future_trajectory_and_event_but_not_true_labels():
    gaps = np.zeros((3, 2, 4)); gaps[1, 0, 1] = .01
    events = np.zeros((3, 2)); events[2, 1] = 1.
    weights = task_weights(gaps, events, .005)
    assert weights[0, 1] > 0 and weights[0, 2] > 0
    np.testing.assert_array_equal(np.diag(weights), 0.)


def test_known_color_summary_and_invalid_predictions():
    rgb = np.array([[[.1, .3, .8], [.8, .3, .1]], [[.1, .3, .8], [.8, .3, .1]]])
    result = rgbd_summary({"rgb": rgb, "depth": np.ones((2, 2)), "object_alpha": np.ones((2, 2))})
    np.testing.assert_allclose(result, [0., .25, 1., .5, .5, .25, 1., .5])
    with pytest.raises(ValueError):
        select_probe([0., .04], np.zeros((2, 2, 1, 1)), np.full((2, 2), np.nan), "guarded", [1.], .02)


def test_extra_probe_and_future_resume_match_continuous_scalar_without_state_reset():
    from dataclasses import asdict
    import copy
    import json
    from pathlib import Path
    from wmfracture.cohesive import InterfaceState
    from wmfracture.cohesive_batch import batch_rollout
    from wmfracture.cohesive_inference import known_action_rollout, reachable_state
    from wmfracture.joint_state import material_model, matched_history
    model = json.loads((Path(__file__).parents[1]/"config/cohesive-probe.json").read_text())
    del model["law"]["fracture_energy"]
    energy = 1.4653
    physical = material_model(model, energy)
    system = reachable_state(physical, matched_history(model, energy, .084))
    release = copy.deepcopy(system.state)
    times = [.052, .056, .072, .104, .156, .196]
    reference = known_action_rollout(system, 0., .196, 5e-6,
        [{"id": "initial", "time": .004, "left": .04, "right": 0.},
         {"id": "probe", "time": .020, "left": .20, "right": 0.},
         {"id": "future", "time": .052, "left": .355, "right": 0.}], times)
    def final_state(result):
        return InterfaceState(**{k: (int(result[k][0, -1]) if k == "step_index" else float(result[k][0, -1])) for k in asdict(release)})
    prefix = batch_rollout(physical, [release], [.04], [0.], 5e-6, [.012])
    extra = batch_rollout(physical, [final_state(prefix)], [0.], [.20], 5e-6, [.016, .020, .026, .032], start_elapsed=.012, future_time=.020)
    future = batch_rollout(physical, [final_state(extra)], [0.], [.355], 5e-6, times, start_elapsed=.032, future_time=.052)
    for key in future:
        np.testing.assert_allclose(future[key][0], [row[key] for row in reference], atol=2e-10, rtol=0)
    assert final_state(extra).time > 1. and final_state(extra).maximum_opening >= release.maximum_opening
