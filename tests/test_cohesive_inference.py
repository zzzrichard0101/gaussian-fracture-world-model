import builtins
import copy
from dataclasses import asdict
import json

import h5py
import numpy as np
import pytest

from wmfracture.cohesive_inference import (Observation, ObservationBundle, accepted_indices, fit_histories,
    known_action_rollout, load_observations, reachable_state, validate_public)


def public_manifest():
    return {"schema": "cohesive-observations-1", "model": {
        "law": {"area": .01, "tensile_stiffness": 1e7, "compressive_stiffness": 1e8,
                "peak_traction": 1000., "fracture_energy": 2.}, "mass_left": 1., "mass_right": 1.,
        "width": .1, "prehistory_duration": 1., "prehistory_samples": 40, "render": {}},
        "observation_times": [0., .004, .008], "observation_end": .008, "observation_actions": [],
        "sensor_noise": {"rgb_sigma": .01, "depth_sigma_m": .0001, "object_alpha_sigma": .01}}


class ScalarRenderer:
    def capture(self, row):
        return {"rgb": np.full((2, 2, 3), .5), "depth": np.full((2, 2), row["x_right"] - row["x_left"]),
                "object_alpha": np.ones((2, 2))}


def test_static_ambiguity_retains_all_candidates_without_reading_files(monkeypatch):
    public = public_manifest()
    renderer = ScalarRenderer()
    system = reachable_state(public["model"], .00185)
    frames = []
    known_action_rollout(system, 0., .008, 1e-5, [], public["observation_times"],
        lambda row: frames.append(Observation(row["elapsed"], **renderer.capture(row))))
    settings = {"candidate_peaks": [0., .001, .002, .003], "dt": 1e-5, "score_slack": .5,
                "rgb_rmse_max": .02, "depth_rmse_m_max": .0002, "object_alpha_rmse_max": .02}
    def forbidden(*args, **kwargs):
        raise AssertionError("Fitting must not open evaluation/private/future files")
    monkeypatch.setattr(builtins, "open", forbidden)
    result = fit_histories(ObservationBundle(public, tuple(frames)), settings, renderer)
    assert result["accepted_indices"] == [0, 1, 2, 3]
    assert [state.state.maximum_opening for state in result["systems"]] == settings["candidate_peaks"]


def test_empty_match_set_is_not_silently_replaced_by_best_candidate():
    metrics = [{"score": 100., "rgb_rmse": 1., "depth_rmse_m": .1, "object_alpha_rmse": .5}]
    settings = {"score_slack": .5, "rgb_rmse_max": .02, "depth_rmse_m_max": .0002, "object_alpha_rmse_max": .02}
    assert accepted_indices(metrics, settings) == []


def test_future_action_or_privileged_metadata_rejected():
    public = public_manifest()
    public["true_peak"] = .002
    with pytest.raises(ValueError, match="privileged"):
        validate_public(public)
    public = public_manifest()
    public["observation_actions"] = [{"id": "future", "time": .012, "left": .2, "right": 0.}]
    with pytest.raises(ValueError, match="future"):
        validate_public(public)


def test_fitter_rejects_future_information_in_settings():
    public = public_manifest()
    settings = {"candidate_peaks": [0., .002], "dt": 1e-5, "score_slack": .5,
                "rgb_rmse_max": .02, "depth_rmse_m_max": .0002, "object_alpha_rmse_max": .02,
                "future_response": [1, 2, 3]}
    with pytest.raises(ValueError, match="privileged inference"):
        fit_histories(ObservationBundle(public, ()), settings, ScalarRenderer())


@pytest.mark.parametrize("contamination", ["future_frame", "state_field"])
def test_public_file_rejects_future_frames_and_hidden_state(tmp_path, contamination):
    public = public_manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(public))
    with h5py.File(tmp_path / "observations.h5", "w") as stream:
        frame = stream.create_group("frames/00000")
        frame.attrs["elapsed"] = .012 if contamination == "future_frame" else 0.
        for name, array in ScalarRenderer().capture({"x_left": -.05, "x_right": .05}).items():
            frame.create_dataset(name, data=array)
        if contamination == "state_field":
            frame.create_dataset("maximum_opening", data=[.002])
    with pytest.raises(ValueError, match="future|Unexpected"):
        load_observations(tmp_path)


def test_forecast_continues_fitted_damage_instead_of_reinitializing():
    model = public_manifest()["model"]
    system = reachable_state(model, .002)
    known_action_rollout(system, 0., .008, 1e-5, [], [0., .008])
    kappa = system.state.maximum_opening
    rows = known_action_rollout(system, .008, .06, 1e-5,
        [{"id": "new-impact", "time": .012, "left": .245, "right": 0.}], [.008, .06])
    assert rows[0]["maximum_opening"] == kappa
    assert rows[-1]["damage"] == 1.
    assert system.state.time == pytest.approx(1.06)
    with pytest.raises(ValueError, match="outside"):
        known_action_rollout(system, .06, .08, 1e-5,
            [{"id": "past", "time": .012, "left": .245, "right": 0.}], [.06, .08])
