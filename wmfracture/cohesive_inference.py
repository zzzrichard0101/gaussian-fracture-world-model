"""Observation-only, bounded history replay baseline for the reduced fixture.

No ground-truth state or future response is an input. Accepted candidates form
a discrete hypothesis set, not a calibrated posterior or confidence interval.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import h5py
import numpy as np

from wmfracture.cohesive import InterfaceLaw, InterfaceState, TwoBlockSystem, prescribed_cycle


FIELDS = {"rgb", "depth", "object_alpha"}
PUBLIC_FIELDS = {"schema", "model", "observation_times", "observation_end", "observation_actions", "sensor_noise"}
MODEL_FIELDS = {"law", "mass_left", "mass_right", "width", "prehistory_duration", "prehistory_samples", "render"}


def tick(value, dt):
    result = round(value / dt)
    if not math.isfinite(value) or value < 0 or abs(result * dt - value) > 1e-10:
        raise ValueError("Time must be a nonnegative integer multiple of dt")
    return result


@dataclass(frozen=True)
class Observation:
    elapsed: float
    rgb: np.ndarray
    depth: np.ndarray
    object_alpha: np.ndarray


@dataclass(frozen=True)
class ObservationBundle:
    public: dict
    frames: tuple


def validate_public(public):
    if set(public) != PUBLIC_FIELDS or public["schema"] != "cohesive-observations-1":
        raise ValueError("Unknown/privileged public manifest fields")
    if set(public["model"]) != MODEL_FIELDS:
        raise ValueError("Unknown/privileged model fields")
    times = public["observation_times"]
    if (not times or times != sorted(set(times)) or times[0] < 0
        or not all(math.isfinite(value) for value in times)
        or times[-1] != public["observation_end"]):
        raise ValueError("Invalid observation times/cutoff")
    for event in public["observation_actions"]:
        if event["time"] >= public["observation_end"]:
            raise ValueError("Observation actions may not contain a future action")
    for key in ("rgb_sigma", "depth_sigma_m", "object_alpha_sigma"):
        if not math.isfinite(public["sensor_noise"][key]) or public["sensor_noise"][key] <= 0:
            raise ValueError("Observation noise scales must be positive")


def load_observations(directory):
    """Only these two public files are read; no parent experiment config is used."""
    directory = Path(directory)
    public = json.loads((directory / "manifest.json").read_text())
    validate_public(public)
    frames = []
    with h5py.File(directory / "observations.h5", "r") as source:
        if set(source) != {"frames"}:
            raise ValueError("Unexpected observation groups; refusing privileged state")
        for key in sorted(source["frames"]):
            group = source["frames"][key]
            if set(group) != FIELDS or set(group.attrs) != {"elapsed"}:
                raise ValueError("Unexpected observation fields/attributes")
            frame = Observation(float(group.attrs["elapsed"]), **{field: group[field][:] for field in FIELDS})
            if (frame.elapsed > public["observation_end"]
                or frame.rgb.shape != (*frame.depth.shape, 3)
                or frame.object_alpha.shape != frame.depth.shape
                or not all(np.isfinite(getattr(frame, field)).all() for field in FIELDS)):
                raise ValueError("Invalid observation frame or future-data contamination")
            frames.append(frame)
    if [frame.elapsed for frame in frames] != public["observation_times"]:
        raise ValueError("Observation file times do not match declared cutoff")
    return ObservationBundle(public, tuple(frames))


def reachable_state(model, peak):
    """Generate a candidate by a complete valid prescribed load/unload history."""
    if not math.isfinite(peak) or peak < 0:
        raise ValueError("History peak must be finite and nonnegative")
    system = TwoBlockSystem(InterfaceLaw(**model["law"]), InterfaceState(-model["width"] / 2, model["width"] / 2),
                            model["mass_left"], model["mass_right"], model["width"])
    prescribed_cycle(system, peak, model["prehistory_samples"], model["prehistory_duration"])
    return system


def known_action_rollout(system, start, stop, dt, actions, sample_times, on_sample=None):
    """Continue the existing state, observe before each tick's known impulse."""
    first, last = tick(start, dt), tick(stop, dt)
    if last < first:
        raise ValueError("Negative rollout interval")
    wanted = {tick(value, dt) for value in sample_times}
    if not wanted or min(wanted) < first or max(wanted) > last:
        raise ValueError("Samples outside rollout interval")
    events, identifiers = {}, set()
    for action in actions:
        if set(action) != {"id", "time", "left", "right"} or action["id"] in identifiers:
            raise ValueError("Invalid/duplicate known action")
        event_tick = tick(action["time"], dt)
        if not first <= event_tick < last:
            raise ValueError("Action outside the current rollout interval")
        identifiers.add(action["id"])
        events.setdefault(event_tick, []).append(action)
    rows = []
    for index in range(first, last + 1):
        if index in wanted:
            row = {"elapsed": index * dt, **asdict(system.state), "opening": system.opening,
                   "damage": system.law.damage(system.state.maximum_opening)}
            rows.append(row)
            if on_sample is not None:
                on_sample(row)
        for event in events.get(index, []):
            system.impulse(event["left"], event["right"])
        if index < last:
            system.advance(dt)
    return rows


def frame_residual(observed, predicted, noise):
    """Foreground RGB, high-alpha depth and full-image soft-mask residuals."""
    observed_mask, predicted_mask = observed.object_alpha > .5, predicted["object_alpha"] > .5
    union = observed_mask | predicted_mask
    common_depth = (observed.object_alpha > .95) & (predicted["object_alpha"] > .95)
    if not union.any() or common_depth.sum() < .5 * max(1, int((observed.object_alpha > .95).sum())):
        raise ValueError("Empty/inadequate visible object support")
    rgb_mse = float(np.mean((observed.rgb[union].astype(float) - predicted["rgb"][union]) ** 2))
    depth_mse = float(np.mean((observed.depth[common_depth].astype(float) - predicted["depth"][common_depth]) ** 2))
    # Exclude distant background so its size cannot hide mask disagreement.
    support = (observed.object_alpha > .01) | (predicted["object_alpha"] > .01)
    alpha_mse = float(np.mean((observed.object_alpha[support].astype(float) - predicted["object_alpha"][support]) ** 2))
    values = {"rgb_rmse": math.sqrt(rgb_mse), "depth_rmse_m": math.sqrt(depth_mse),
              "object_alpha_rmse": math.sqrt(alpha_mse)}
    values["score"] = (rgb_mse / noise["rgb_sigma"] ** 2 + depth_mse / noise["depth_sigma_m"] ** 2
                       + alpha_mse / noise["object_alpha_sigma"] ** 2) / 3
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite observation residual")
    return values


def accepted_indices(metrics, settings):
    if not metrics:
        return []
    best = min(row["score"] for row in metrics)
    return [index for index, row in enumerate(metrics)
            if row["score"] <= best + settings["score_slack"]
            and row["rgb_rmse"] <= settings["rgb_rmse_max"]
            and row["depth_rmse_m"] <= settings["depth_rmse_m_max"]
            and row["object_alpha_rmse"] <= settings["object_alpha_rmse_max"]]


def fit_histories(bundle, settings, renderer):
    """Pure observation fit. No file IO, future schedule, labels or evaluator."""
    validate_public(bundle.public)
    if set(settings) != {"candidate_peaks", "dt", "score_slack", "rgb_rmse_max", "depth_rmse_m_max", "object_alpha_rmse_max"}:
        raise ValueError("Unknown/privileged inference settings")
    if [frame.elapsed for frame in bundle.frames] != bundle.public["observation_times"]:
        raise ValueError("Frame times/cutoff mismatch")
    peaks = settings["candidate_peaks"]
    if not peaks or len(set(peaks)) != len(peaks):
        raise ValueError("Candidate peaks must be distinct")
    metrics, systems = [], []
    for peak in peaks:
        system = reachable_state(bundle.public["model"], peak)
        frame_metrics = []
        def observe(row):
            frame = bundle.frames[len(frame_metrics)]
            frame_metrics.append(frame_residual(frame, renderer.capture(row), bundle.public["sensor_noise"]))
        known_action_rollout(system, 0., bundle.public["observation_end"], settings["dt"],
                             bundle.public["observation_actions"], bundle.public["observation_times"], observe)
        average = {key: float(np.mean([row[key] for row in frame_metrics])) for key in frame_metrics[0]}
        average["frame_metrics"] = frame_metrics
        metrics.append(average)
        systems.append(system)
    selected = accepted_indices(metrics, settings)
    return {"metrics": metrics, "systems": systems, "accepted_indices": selected,
            "best_index": min(range(len(metrics)), key=lambda index: metrics[index]["score"]),
            "forecasting_allowed": bool(selected)}
