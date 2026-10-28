"""Generate grouped supervised data and a shared physics bank in the GS env."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from cohesive_probe import BlockObservations
from wmfracture.action_data import assert_group_split, prefix_features
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.cohesive_inference import Observation, accepted_indices, frame_residual, reachable_state, known_action_rollout


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def generate(root):
    start = time.perf_counter()
    config = json.loads((root/"config.json").read_text())
    base = json.loads((Path(os.environ["WM_PROJECT_ROOT"])/config["base_model_config"]).read_text())
    model = {key: base[key] for key in ("law", "mass_left", "mass_right", "width", "prehistory_duration", "prehistory_samples", "render")}
    model["render"]["width"] = config["render_width"]
    observer = BlockObservations(model)
    for directory in ("prefixes", "inputs", "supervision", "private"):
        (root/directory).mkdir()
    belief_mode = config.get("write_belief_data", False)
    if belief_mode:
        (root/"belief-inputs").mkdir()
        (root/"state-supervision").mkdir()
    rng = np.random.default_rng(config["seed"])
    peaks = np.linspace(config["history_peak_min_m"], config["history_peak_max_m"], sum(config["history_counts"].values()))
    peaks = peaks[rng.permutation(len(peaks))]
    if any(np.isclose(value, config["candidate_peaks"], atol=1e-12, rtol=0).any() for value in peaks):
        raise ValueError("Exact supervised history occurs in candidate bank")
    groups, histories, offset = {}, {}, 0
    for split, count in config["history_counts"].items():
        groups[split] = []
        for index in range(offset, offset+count):
            key = f"history-{index:03d}"
            histories[key] = float(peaks[index])
            groups[split].append(key)
        offset += count
    assert_group_split(groups)
    write_json(root/"split-groups.json", groups)
    write_json(root/"private/history-labels.json", histories)
    write_json(root/"known-model.json", model)
    actions = sorted(set(value for values in config["actions"].values() for value in values))
    times = sorted(set(config["context_times"]+config["target_times"]))
    context_indices = [times.index(value) for value in config["context_times"]]
    target_indices = [times.index(value) for value in config["target_times"]]
    settings = {**config["acceptance"], "candidate_peaks": config["candidate_peaks"], "dt": config["candidate_dt"]}
    # Generate candidate bank once per known probe and future action.
    candidate_states, candidate_probes, candidate_actions = [], [], []
    bank_meta = []
    for probe in config["probe_velocity_increments"]:
        for impact in actions:
            bank_meta.append({"probe": probe, "impact": impact})
            for peak in config["candidate_peaks"]:
                candidate_states.append(reachable_state(model, peak).state)
                candidate_probes.append(probe); candidate_actions.append(impact)
    candidate = batch_rollout(model, candidate_states, candidate_probes, candidate_actions,
        config["candidate_dt"], times, config["probe_time"], config["future_time"])
    nprior = len(config["candidate_peaks"])
    bank_gaps = candidate["opening"][:, target_indices].reshape(len(bank_meta), nprior, -1)
    np.savez_compressed(root/"physics-bank.npz", gaps=bank_gaps,
        failure=candidate["damage"][:, -1].reshape(len(bank_meta), nprior) == 1.)
    write_json(root/"physics-bank-index.json", bank_meta)
    candidate_frames = {}
    cutoff_states = []
    for probe in config["probe_velocity_increments"]:
        bank_index = next(i for i, item in enumerate(bank_meta) if item["probe"] == probe)
        frames = []
        for prior in range(nprior):
            row_index = bank_index*nprior+prior
            frames.append([observer.capture({"x_left": candidate["x_left"][row_index, t],
                                             "x_right": candidate["x_right"][row_index, t]}) for t in context_indices])
        candidate_frames[probe] = frames
        if belief_mode:
            for peak in config["candidate_peaks"]:
                system = reachable_state(model, peak)
                known_action_rollout(system, 0., max(config["context_times"]), config["candidate_dt"],
                    [{"id": "observed-probe", "time": config["probe_time"], "left": probe, "right": 0.}],
                    [max(config["context_times"])])
                cutoff_states.append({"probe": probe, "history_peak": peak, "state": asdict(system.state)})
    if belief_mode:
        write_json(root/"candidate-prefix-states.json", {"schema": "reduced-reachable-cutoff-states-1",
            "observation_end": max(config["context_times"]), "model": model, "candidates": cutoff_states})
    prefix_records, sample_records = [], []
    states, probes, impacts = [], [], []
    # No future action takes place before the prefix cutoff.
    if not config["probe_time"] <= max(config["context_times"]) < config["future_time"]:
        raise ValueError("Intervention clocks overlap the observation cutoff")
    for split, group_ids in groups.items():
        for group in group_ids:
            for probe in config["probe_velocity_increments"]:
                prefix_index = len(prefix_records)
                # Replay the complete prescribed prehistory, never set damage directly.
                release = reachable_state(model, histories[group])
                observed = batch_rollout(model, [release.state], [probe], [0.], config["truth_dt"],
                    config["context_times"], config["probe_time"], config["future_time"])
                # Same exact noisy static observation for all histories: no spurious identification signal.
                noise_rng = np.random.default_rng(config["seed"] if probe == 0 else config["seed"]+1000+prefix_index)
                observed_frames = []
                for t, elapsed in enumerate(config["context_times"]):
                    fields = observer.capture({"x_left": observed["x_left"][0, t], "x_right": observed["x_right"][0, t]})
                    noise = config["sensor_noise"]
                    observed_frames.append(Observation(elapsed,
                        np.clip(fields["rgb"]+noise_rng.normal(0, noise["rgb_sigma"], fields["rgb"].shape), 0, 1).astype(np.float32),
                        (fields["depth"]+noise_rng.normal(0, noise["depth_sigma_m"], fields["depth"].shape)).astype(np.float32),
                        np.clip(fields["object_alpha"]+noise_rng.normal(0, noise["object_alpha_sigma"], fields["object_alpha"].shape), 0, 1).astype(np.float32)))
                rgb, depth, alpha = [np.stack([getattr(frame, key) for frame in observed_frames]) for key in ("rgb", "depth", "object_alpha")]
                np.savez_compressed(root/"prefixes"/f"prefix-{prefix_index:03d}.npz", rgb=rgb, depth=depth, alpha=alpha)
                metrics = []
                for prior_frames in candidate_frames[probe]:
                    frame_metrics = [frame_residual(frame, generated, config["sensor_noise"])
                        for frame, generated in zip(observed_frames, prior_frames)]
                    metrics.append({key: float(np.mean([item[key] for item in frame_metrics])) for key in frame_metrics[0]})
                accepted = np.zeros(nprior, dtype=bool)
                accepted[accepted_indices(metrics, settings)] = True
                best = int(np.argmin([item["score"] for item in metrics]))
                features = np.concatenate([prefix_features(rgb, depth, alpha), np.log1p([item["score"] for item in metrics])]).astype(np.float32)
                prefix_records.append({"index": prefix_index, "split": split, "history_group": group, "probe": probe})
                if belief_mode:
                    prefix_records[-1].update({"features": features, "scores": [item["score"] for item in metrics], "accepted": accepted})
                tasks = [("train", config["actions"]["train"])] if split == "train" else (
                    [("validation", config["actions"]["validation"])] if split == "validation" else
                    [(name, config["actions"][name]) for name in ("test_seen", "test_interpolation", "test_extrapolation")])
                for regime, future_values in tasks:
                    for impact in future_values:
                        bank_index = next(i for i, item in enumerate(bank_meta) if item == {"probe": probe, "impact": impact})
                        sample_records.append({"split": split, "regime": regime, "prefix_index": prefix_index,
                            "history_group": group, "probe": probe, "impact": impact, "features": features,
                            "accepted": accepted, "rgbd_best": best, "bank_index": bank_index})
                        states.append(type(release.state)(**asdict(release.state)))
                        probes.append(probe); impacts.append(impact)
                if (prefix_index+1) % 8 == 0:
                    print(f"PREFIXES {prefix_index+1}/{2*len(histories)}", flush=True)
    truth = batch_rollout(model, states, probes, impacts, config["truth_dt"], times,
        config["probe_time"], config["future_time"])
    write_json(root/"prefix-index.json", [{key: value for key, value in record.items() if key not in ("features", "scores", "accepted")} for record in prefix_records])
    if belief_mode:
        from wmfracture.belief_state import state_targets
        for split in groups:
            records = [record for record in prefix_records if record["split"] == split]
            np.savez_compressed(root/"belief-inputs"/f"{split}.npz",
                prefix_index=np.array([r["index"] for r in records]), scores=np.array([r["scores"] for r in records]),
                accepted=np.stack([r["accepted"] for r in records]), rgbd_features=np.stack([r["features"] for r in records]),
                probe=np.array([[r["probe"]] for r in records], dtype=np.float32))
            target_dir = root/("private" if split == "test" else "state-supervision")
            np.savez_compressed(target_dir/f"{split}-states.npz", prefix_index=np.array([r["index"] for r in records]),
                targets=state_targets([histories[r["history_group"]] for r in records], config["candidate_peaks"]))
    for split in groups:
        indices = [i for i, item in enumerate(sample_records) if item["split"] == split]
        selected = [sample_records[i] for i in indices]
        np.savez_compressed(root/"inputs"/f"{split}.npz",
            rgbd_features=np.stack([item["features"] for item in selected]),
            prefix_index=np.array([item["prefix_index"] for item in selected]),
            actions=np.array([[item["probe"], item["impact"]] for item in selected], dtype=np.float32),
            accepted=np.stack([item["accepted"] for item in selected]), rgbd_best=np.array([item["rgbd_best"] for item in selected]),
            bank_index=np.array([item["bank_index"] for item in selected]))
        # Training code has labeled train/validation archives, not test labels.
        label_directory = "private" if split == "test" else "supervision"
        np.savez_compressed(root/label_directory/f"{split}-targets.npz",
            gaps=truth["opening"][indices][:, target_indices].astype(np.float32),
            failure=(truth["damage"][indices, -1] == 1.).astype(np.float32))
        write_json(root/"inputs"/f"{split}-index.json", [{key: item[key] for key in
            ("regime", "prefix_index", "history_group", "probe", "impact")} for item in selected])
    write_json(root/"data-summary.json", {"prefixes": len(prefix_records), "samples": len(sample_records),
        "history_counts": config["history_counts"], "wall_seconds": time.perf_counter()-start,
        "truth_dt": config["truth_dt"], "candidate_dt": config["candidate_dt"],
        "candidate_histories": nprior, "history_groups_disjoint": True,
        "stationary_prefixes_identical_by_construction": True})
    print(f"DATA_READY samples={len(sample_records)} seconds={time.perf_counter()-start:.1f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    generate(parser.parse_args().run)
