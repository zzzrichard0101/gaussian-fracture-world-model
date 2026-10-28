import numpy as np
from wmfracture.separation import (spatial_components, ComponentTracker,
                                   persistent_motion_candidate, simultaneous_consensus)


def test_spatial_groups_ignore_dust_and_respect_gap_threshold():
    points = np.array([[0, 0, 0], [.1, 0, 0], [1., 0, 0], [1.1, 0, 0], [5, 0, 0]], dtype=float)
    volumes = np.array([1., 1., 1., 1., .001])
    velocities = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    labels, result = spatial_components(points, volumes, velocities, .2)
    assert len(result["major_components"]) == 2
    assert result["dust_mass_fraction"] < .001
    np.testing.assert_allclose(result["two_largest_gap"], .9)
    np.testing.assert_allclose(result["two_largest_relative_speed"], 1.)
    _, joined = spatial_components(points, volumes, velocities, 1.01)
    assert len(joined["major_components"]) == 1


def test_tracker_uses_particle_identity_and_restarts_after_merge():
    tracker = ComponentTracker(np.ones(4))
    first = [{"component": 0}, {"component": 1}]
    tracker.update(np.array([0, 0, 1, 1]), first, .1)
    second = [{"component": 8}, {"component": 9}]
    tracker.update(np.array([9, 9, 8, 8]), second, .16)
    np.testing.assert_allclose([g["consecutive_duration"] for g in second], [.06, .06])
    assert second[0]["track_id"] == first[1]["track_id"]
    tracker.update(np.array([0, 0, 0, 0]), [{"component": 0}], .2)
    split = [{"component": 0}, {"component": 1}]
    tracker.update(np.array([0, 0, 1, 1]), split, .3)
    assert all(g["consecutive_duration"] == 0.0 for g in split)


def test_positive_control_detects_two_persistent_moving_blocks():
    # Prescribed kinematic separation validates the detector, not fracture physics.
    axis = np.arange(3) * .01
    left = np.stack(np.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3)
    right = left + [.03, 0, 0]
    initial = np.concatenate([left, right])
    velocity = np.zeros_like(initial)
    velocity[:len(left), 0], velocity[len(left):, 0] = -.5, .5
    volumes = np.ones(len(initial))
    rows = {}
    for radius in (.02, .03, .04):
        tracker = ComponentTracker(volumes)
        observations = []
        for index, timestamp in enumerate([0., .03, .06, .09, .12]):
            labels, diagnostics = spatial_components(initial + velocity * timestamp, volumes, velocity, radius)
            tracker.update(labels, diagnostics["major_components"], timestamp)
            observations.append({"frame": index, "separation_candidate": persistent_motion_candidate(
                diagnostics["major_components"], diagnostics["two_largest_relative_speed"])})
        assert not observations[0]["separation_candidate"]
        assert observations[-1]["separation_candidate"]
        rows[str(radius)] = observations
    assert 4 in simultaneous_consensus(rows, {key: True for key in rows})


def test_consensus_requires_same_observation_time():
    rows = {"small": [{"frame": 1, "separation_candidate": True}, {"frame": 2, "separation_candidate": False}],
            "large": [{"frame": 1, "separation_candidate": False}, {"frame": 2, "separation_candidate": True}]}
    assert simultaneous_consensus(rows, {"small": True, "large": True}) == []
