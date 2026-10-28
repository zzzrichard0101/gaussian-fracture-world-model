from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from wmfracture.history_refinement import initial_points, propose, selection, validate_search
from wmfracture.cohesive import InterfaceState, InterfaceLaw, TwoBlockSystem
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.cohesive_inference import reachable_state, known_action_rollout


def settings():
    return json.loads((Path(__file__).parents[1]/"config/history-refinement.json").read_text())["search"]


def test_refinement_is_observation_only_and_obeys_exact_distinct_evaluation_budget():
    cfg = settings()
    for method in ("greedy49", "multiregion49"):
        points = initial_points(method, cfg)
        while len(points) < cfg["budget"]:
            scores = [(value-.00137)**2 for value in points]
            extra = propose(points, scores, method, cfg)
            assert len(set(points+extra)) == len(points)+len(extra)
            points.extend(extra)
        assert len(points) == 49
        assert min(abs(value-.00137) for value in points) < 1e-6
    with pytest.raises(ValueError, match="exclude future"):
        validate_search(dict(cfg, future_response=[0.]))


def test_equal_observations_explore_global_domain_and_refuse_unique_state():
    cfg = settings()
    points = initial_points("multiregion49", cfg)
    while len(points) < 49:
        points.extend(propose(points, np.ones(len(points)), "multiregion49", cfg))
    np.testing.assert_allclose(sorted(points), initial_points("uniform49", cfg), atol=1e-14)
    metrics = [{"score": 1., "rgb_rmse": .001, "depth_rmse_m": 1e-5, "object_alpha_rmse": .001}]*49
    result = selection(metrics, cfg)
    assert result["status"] == "ambiguous" and result["point_index"] is None
    assert len(result["accepted_indices"]) == 49


def test_multiregion_proposals_cover_distinct_observed_basins():
    cfg = settings()
    points = initial_points("multiregion49", cfg)
    scores = [min(((x-.0005)/.00025)**2, .1+((x-.0025)/.00025)**2) for x in points]
    extra = propose(points, scores, "multiregion49", cfg)
    assert any(.00025 < value < .00075 for value in extra)
    assert any(.00225 < value < .00275 for value in extra)


def test_batch_cutoff_full_state_can_resume_without_erasing_work_velocity_or_clock():
    config = json.loads((Path(__file__).parents[1]/"config/cohesive-probe.json").read_text())
    system = reachable_state(config, .00137)
    before = asdict(system.state)
    result = batch_rollout(config, [system.state], [.04], [0.], 5e-6, [0., .004, .008, .012])
    assert asdict(system.state) == before
    state = InterfaceState(**{key: (int(result[key][0, -1]) if key == "step_index" else float(result[key][0, -1])) for key in before})
    resumed = TwoBlockSystem(InterfaceLaw(**config["law"]), state, config["mass_left"], config["mass_right"], config["width"])
    known_action_rollout(system, 0., .012, 5e-6, [{"id": "p", "time": .004, "left": .04, "right": 0.}], [.012])
    assert resumed.state.maximum_opening >= before["maximum_opening"]
    for key in before:
        assert getattr(resumed.state, key) == pytest.approx(getattr(system.state, key), abs=2e-10)
    future = [{"id": "f", "time": .036, "left": .255, "right": 0.}]
    first = known_action_rollout(resumed, .012, .18, 5e-6, future, [.036, .04, .18])
    second = known_action_rollout(system, .012, .18, 5e-6, future, [.036, .04, .18])
    np.testing.assert_allclose([r["opening"] for r in first], [r["opening"] for r in second], atol=2e-10)
