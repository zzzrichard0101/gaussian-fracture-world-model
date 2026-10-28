"""Old Gaussian/CUDA environment: export prefix-only learned-prior inputs.

Real suffix and physical reference remain in a separate evaluation directory.
No original eight-frame candidate selection is reused for the four-frame task.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from cohesive_probe import BlockObservations
from wmfracture.cohesive_inference import (accepted_indices, frame_residual, known_action_rollout,
                                         load_observations, reachable_state, tick)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    config = json.loads((root / "config.json").read_text())
    source = Path(config["source_experiment"])
    settings = json.loads((source / "fit-settings.json").read_text())
    identifiers = json.loads((source / "dataset-index.json").read_text())
    for identifier in identifiers:
        public = source / "samples" / identifier / "public"
        bundle = load_observations(public)
        model = bundle.public["model"]
        if model["render"]["width"] != config["image_size"] or len(bundle.frames) != config["observation_slots"]:
            raise ValueError("Source observation geometry/length mismatch")
        prefix = bundle.frames[:config["context_slots"]]
        if any(prefix[-1].elapsed < action["time"] <= bundle.public["observation_end"]
               for action in bundle.public["observation_actions"]):
            raise ValueError("Latent forecast cannot condition on an intervention outside the observed prefix")
        context_rgb = np.stack([frame.rgb for frame in prefix])
        context_roi = np.stack([frame.object_alpha for frame in prefix]).max(0) > .5
        future = json.loads((source / "samples" / identifier / "future-input.json").read_text())
        if any(action["time"] <= bundle.public["observation_end"] for action in future["actions"]):
            raise ValueError("Learned latent horizon must not contain a new unconditioned intervention")
        observer = BlockObservations(model)
        folder = root / "inputs" / identifier
        evaluation = root / "evaluation" / identifier
        physics = root / "physics-candidates" / identifier
        folder.mkdir(parents=True); evaluation.mkdir(parents=True); physics.mkdir(parents=True)
        clips, metrics = [], []
        interval = future["sample_interval"]
        sample_times = [i * interval for i in range(tick(future["end"], interval) + 1)]
        observation_indices = {tick(frame.elapsed, interval): index for index, frame in enumerate(bundle.frames)}
        for index, peak in enumerate(settings["candidate_peaks"]):
            system = reachable_state(model, peak)
            clip, residuals = [], []
            def capture(row):
                frame_index = observation_indices.get(tick(row["elapsed"], interval))
                if frame_index is None:
                    return
                fields = observer.capture(row)
                clip.append(fields["rgb"])
                if frame_index < config["context_slots"]:
                    residuals.append(frame_residual(prefix[frame_index], fields, bundle.public["sensor_noise"]))
            rows = known_action_rollout(system, 0., future["end"], settings["dt"],
                [*bundle.public["observation_actions"], *future["actions"]], sample_times, capture)
            values = np.stack(clip)
            values[:len(prefix)] = context_rgb
            clips.append(values)
            metrics.append({key: float(np.mean([record[key] for record in residuals])) for key in residuals[0]})
            with h5py.File(physics / f"candidate-{index:02d}.h5", "x") as stream:
                for key in rows[0]:
                    stream.create_dataset(key, data=np.array([row[key] for row in rows]))
        accepted = accepted_indices(metrics, settings)
        np.savez_compressed(folder / "rank-input.npz", context_rgb=context_rgb, candidate_rgb=np.stack(clips),
            context_roi=context_roi.astype(np.float32),
            prefix_metrics=np.array([[row[key] for key in ("score", "rgb_rmse", "depth_rmse_m", "object_alpha_rmse")] for row in metrics]))
        write_json(folder / "manifest.json", {"schema": "vjepa-physics-prefix-input-1", "sample": identifier,
            "candidate_peaks": settings["candidate_peaks"], "rgbd_accepted_indices": accepted,
            "rgbd_best_index": int(np.argmin([row["score"] for row in metrics])),
            "context_times": [frame.elapsed for frame in prefix],
            "latent_horizon_times": [frame.elapsed for frame in bundle.frames[len(prefix):]],
            "model": model, "future_input": future,
            "rank_arrays_sha256": hashlib.sha256((folder / "rank-input.npz").read_bytes()).hexdigest(),
            "scope": "Only prefix RGB-D residuals and simulated candidates; no real suffix or truth states."})
        np.savez_compressed(evaluation / "real-observed-clip.npz", rgb=np.stack([frame.rgb for frame in bundle.frames]))
        # Copy only for the later evaluator, never included in the rank archive.
        import shutil
        shutil.copy2(source / "samples" / identifier / "private/evaluation.h5", evaluation / "truth.h5")
        write_json(evaluation / "metadata.json", {"protocol": "fixed-small-impulse" if bundle.public["observation_actions"] else "stationary",
            "source": str(public), "future_response_for_fit": False})
        print(f"EXPORTED {identifier} prefix={len(prefix)} accepted={len(accepted)}/{len(clips)}", flush=True)
    write_json(root / "dataset-index.json", identifiers)


if __name__ == "__main__":
    main()
