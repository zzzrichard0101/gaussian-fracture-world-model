"""Generate -> observation-only fit -> sealed future rollout -> evaluation.

Separate processes and narrow input files keep private labels/future responses
out of candidate selection. This is a finite history-search baseline on the
reduced cohesive fixture, not a learned world model or a novel inference rule.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

from wmfracture.cohesive import load_state, save_state
from wmfracture.cohesive_inference import (fit_histories, known_action_rollout, load_observations,
                                         reachable_state, tick)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid_times(start, end, interval):
    return [index * interval for index in range(tick(start, interval), tick(end, interval) + 1)]


def write_rows(path, rows):
    with h5py.File(path, "x") as stream:
        for key in rows[0]:
            stream.create_dataset(key, data=np.array([row[key] for row in rows]))


def read_rows(path):
    with h5py.File(path, "r") as stream:
        arrays = {key: stream[key][:] for key in stream}
    return [{key: value[index].item() for key, value in arrays.items()} for index in range(len(arrays["elapsed"]))]


def generate(root):
    from cohesive_probe import BlockObservations
    config = json.loads((root / "private-experiment.json").read_text())
    project = Path(os.environ["WM_PROJECT_ROOT"])
    base = json.loads((project / config["base_model_config"]).read_text())
    model = {key: base[key] for key in ("law", "mass_left", "mass_right", "width", "prehistory_duration", "prehistory_samples", "render")}
    model["render"]["width"] = config["render_width"]
    observer = BlockObservations(model)
    sample_ids = []
    for truth_index, peak in enumerate(config["truth_preload_peaks"]):
        if peak in config["candidate_peaks"]:
            raise ValueError("Exact test history occurs in the fixed candidate bank")
        for informative in (False, True):
            identifier = f"sample-{len(sample_ids):02d}"
            sample_ids.append(identifier)
            folder = root / "samples" / identifier
            public_dir, private_dir = folder / "public", folder / "private"
            public_dir.mkdir(parents=True); private_dir.mkdir()
            actions = [config["probe_action"]] if informative else []
            public = {"schema": "cohesive-observations-1", "model": model,
                      "observation_times": config["observation_times"],
                      "observation_end": max(config["observation_times"]),
                      "observation_actions": actions, "sensor_noise": config["sensor_noise"]}
            write_json(public_dir / "manifest.json", public)
            write_json(folder / "future-input.json", {"schema": "cohesive-future-input-1",
                "actions": config["future_actions"], "end": config["forecast_end"],
                "sample_interval": config["sample_interval"]})
            write_json(private_dir / "truth.json", {"preload_peak": peak, "truth_index": truth_index,
                "observation_protocol": "fixed-small-impulse" if informative else "stationary",
                "truth_dt": config["truth_dt"], "paired_noise_seed": config["noise_seed"] + truth_index})
            system = reachable_state(model, peak)
            rng = np.random.default_rng(config["noise_seed"] + truth_index)
            observation_indices = {tick(value, config["sample_interval"]): index
                                   for index, value in enumerate(config["observation_times"])}
            with h5py.File(public_dir / "observations.h5", "x") as observations:
                def capture(row):
                    sample_index = tick(row["elapsed"], config["sample_interval"])
                    if sample_index not in observation_indices:
                        return
                    fields = observer.capture(row)
                    group = observations.create_group(f"frames/{observation_indices[sample_index]:05d}")
                    group.attrs["elapsed"] = config["observation_times"][observation_indices[sample_index]]
                    noise = config["sensor_noise"]
                    rgb = np.clip(fields["rgb"] + rng.normal(0, noise["rgb_sigma"], fields["rgb"].shape), 0, 1).astype(np.float32)
                    depth = (fields["depth"] + rng.normal(0, noise["depth_sigma_m"], fields["depth"].shape)).astype(np.float32)
                    alpha = np.clip(fields["object_alpha"] + rng.normal(0, noise["object_alpha_sigma"], fields["object_alpha"].shape), 0, 1).astype(np.float32)
                    for key, value in (("rgb", rgb), ("depth", depth), ("object_alpha", alpha)):
                        group.create_dataset(key, data=value, compression="gzip", compression_opts=1)
                rows = known_action_rollout(system, 0., config["forecast_end"], config["truth_dt"],
                    [*actions, *config["future_actions"]], grid_times(0., config["forecast_end"], config["sample_interval"]), capture)
            write_rows(private_dir / "evaluation.h5", rows)
            print(f"generated {identifier} public_frames={len(config['observation_times'])}", flush=True)
    write_json(root / "dataset-index.json", sample_ids)


def infer(root, identifier):
    from cohesive_probe import BlockObservations
    folder = root / "samples" / identifier
    output = folder / "fit"
    output.mkdir()
    (output / "states").mkdir()
    settings_path = root / "fit-settings.json"
    settings = json.loads(settings_path.read_text())
    bundle = load_observations(folder / "public")
    observer = BlockObservations(bundle.public["model"])
    started = time.perf_counter()
    fitted = fit_histories(bundle, settings, observer)
    records = []
    for index, (peak, state, metrics) in enumerate(zip(settings["candidate_peaks"], fitted["systems"], fitted["metrics"])):
        relative = f"states/candidate-{index:02d}.json"
        save_state(output / relative, state)
        records.append({"index": index, "prehistory_peak": peak, "metrics": metrics,
                        "accepted": index in fitted["accepted_indices"], "state": relative,
                        "state_sha256": digest(output / relative)})
    selection = {"schema": "cohesive-history-selection-1", "status": "matched" if fitted["forecasting_allowed"] else "no_match",
        "observation_end": bundle.public["observation_end"], "model": bundle.public["model"],
        "inference_dt": settings["dt"], "best_index": fitted["best_index"],
        "accepted_indices": fitted["accepted_indices"], "candidates": records,
        "inputs_read": {"manifest": digest(folder / "public/manifest.json"),
                        "observations": digest(folder / "public/observations.h5"), "fit_settings": digest(settings_path)},
        "future_action_read_during_fit": False, "future_response_read_during_fit": False,
        "private_state_read_during_fit": False, "fit_wall_seconds": time.perf_counter() - started,
        "forward_observation_calls": len(records), "observation_renders": len(records) * len(bundle.frames),
        "note": "Heuristic discrete candidate set, no calibrated probability and no fallback when empty."}
    path = output / "selection.json"
    write_json(path, selection)
    (output / "selection.sha256").write_text(digest(path))
    print(f"fit {identifier} accepted={len(fitted['accepted_indices'])}/{len(records)} best={fitted['best_index']}", flush=True)


def sealed_selection(folder):
    path = folder / "fit/selection.json"
    seal = (folder / "fit/selection.sha256").read_text().strip()
    if digest(path) != seal:
        raise ValueError("Candidate selection changed after sealing")
    return json.loads(path.read_text()), seal


def forecast(root, identifier):
    folder = root / "samples" / identifier
    selection, seal = sealed_selection(folder)
    future = json.loads((folder / "future-input.json").read_text())
    if set(future) != {"schema", "actions", "end", "sample_interval"} or future["schema"] != "cohesive-future-input-1":
        raise ValueError("Invalid future input; response/state fields prohibited")
    output = folder / "prediction"
    output.mkdir()
    summaries = []
    for record in selection["candidates"]:
        path = folder / "fit" / record["state"]
        if digest(path) != record["state_sha256"]:
            raise ValueError("Fitted candidate state was changed")
        system = load_state(path)
        rows = known_action_rollout(system, selection["observation_end"], future["end"], selection["inference_dt"],
            future["actions"], grid_times(selection["observation_end"], future["end"], future["sample_interval"]))
        target = f"candidate-{record['index']:02d}.h5"
        write_rows(output / target, rows)
        failed = next((row["elapsed"] for row in rows if row["damage"] == 1.), None)
        summaries.append({"index": record["index"], "accepted": record["accepted"], "trajectory": target,
            "final_gap_m": rows[-1]["opening"], "fully_separated": rows[-1]["damage"] == 1.,
            "first_failure_sample_elapsed": failed})
    write_json(output / "prediction.json", {"schema": "cohesive-hypothesis-forecast-1", "selection_sha256": seal,
        "status": "candidate_set_forecast" if selection["accepted_indices"] else "no_observation_match",
        "future_input_sha256": digest(folder / "future-input.json"), "candidates": summaries,
        "note": "All candidates rolled once for shared-budget prior/default/selected baselines; no state averaging or probability calibration."})
    print(f"forecast {identifier} candidates={len(summaries)} selection_seal={seal[:12]}", flush=True)


def evaluate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    from cohesive_probe import BlockObservations
    identifiers = json.loads((root / "dataset-index.json").read_text())
    results = []
    figure, axes = plt.subplots((len(identifiers) + 1) // 2, 2, figsize=(12, 7.8), constrained_layout=True)
    for axis in list(axes.flat)[len(identifiers):]:
        axis.set_visible(False)
    for identifier, axis in zip(identifiers, axes.flat):
        folder = root / "samples" / identifier
        selection, seal = sealed_selection(folder)
        prediction = json.loads((folder / "prediction/prediction.json").read_text())
        if prediction["selection_sha256"] != seal:
            raise ValueError("Forecast does not correspond to sealed observation selection")
        truth_meta = json.loads((folder / "private/truth.json").read_text())
        truth_all = read_rows(folder / "private/evaluation.h5")
        trajectories = [read_rows(folder / "prediction" / record["trajectory"]) for record in prediction["candidates"]]
        elapsed = np.array([row["elapsed"] for row in trajectories[0]])
        truth = [row for row in truth_all if row["elapsed"] >= selection["observation_end"] - 1e-12]
        if len(truth) != len(elapsed) or not np.allclose([row["elapsed"] for row in truth], elapsed, atol=1e-12, rtol=0):
            raise ValueError("Prediction and reference sampling times differ")
        gaps = np.array([[row["opening"] for row in rows] for rows in trajectories])
        actual = np.array([row["opening"] for row in truth])
        future_input = json.loads((folder / "future-input.json").read_text())
        future_mask = elapsed > min(event["time"] for event in future_input["actions"])
        accepted = selection["accepted_indices"]
        best = selection["best_index"]
        default = next(row["index"] for row in selection["candidates"] if row["prehistory_peak"] == 0.)
        rmse = lambda values: float(np.sqrt(np.mean((values[future_mask] - actual[future_mask]) ** 2)))
        prior_width = gaps.max(0) - gaps.min(0)
        result = {"sample": identifier, "truth_preload_peak_evaluation_only": truth_meta["preload_peak"],
            "observation_protocol": truth_meta["observation_protocol"], "candidate_count": len(trajectories),
            "accepted_count": len(accepted), "accepted_history_peaks": [selection["candidates"][index]["prehistory_peak"] for index in accepted],
            "best_history_peak": selection["candidates"][best]["prehistory_peak"],
            "true_history_excluded_from_bank": truth_meta["preload_peak"] not in [row["prehistory_peak"] for row in selection["candidates"]],
            "best_single_future_gap_rmse_m": rmse(gaps[best]), "history_blind_future_gap_rmse_m": rmse(gaps[default]),
            "prior_mean_future_gap_width_m": float(prior_width[future_mask].mean()),
            "actual_fully_separated": truth[-1]["damage"] == 1.,
            "selection_seal_verified": True, "future_response_used_for_fitting": False}
        axis.fill_between(elapsed, gaps.min(0) * 1000, gaps.max(0) * 1000, color="gray", alpha=.15, label="All prior candidates")
        if accepted:
            low, high = gaps[accepted].min(0), gaps[accepted].max(0)
            separated = [prediction["candidates"][index]["fully_separated"] for index in accepted]
            result.update(selected_mean_future_gap_width_m=float((high-low)[future_mask].mean()),
                selected_envelope_coverage_fraction=float(((actual >= low-1e-9) & (actual <= high+1e-9))[future_mask].mean()),
                selected_failure_outcome="mixed" if len(set(separated)) > 1 else "fully_separated" if separated[0] else "bonded_or_partly_damaged")
            axis.fill_between(elapsed, low * 1000, high * 1000, color="#278f9d", alpha=.3, label="Observation-matched set")
        axis.plot(elapsed, actual * 1000, color="black", label="Reference, evaluation only")
        axis.plot(elapsed, gaps[best] * 1000, color="#1478b7", linestyle="--", label="Best single candidate")
        axis.plot(elapsed, gaps[default] * 1000, color="#b46445", linestyle=":", label="Default undamaged history")
        coverage_label = (f"Time-sample envelope coverage: {result['selected_envelope_coverage_fraction']:.1%} (uncalibrated)"
                          if accepted else "No observation-matched candidate")
        axis.set_title(f"{identifier}: {truth_meta['observation_protocol']} | {len(accepted)}/{len(trajectories)} retained\n{coverage_label}", fontsize=10)
        axis.set_xlabel("Time after release (s)"); axis.set_ylabel("Interface gap (mm)")
        axis.grid(alpha=.2); axis.legend(fontsize=7)
        results.append(result)
        if identifier == identifiers[-1]:
            observer = BlockObservations(selection["model"])
            width = selection["model"]["render"]["width"]
            video = root / "prediction-frames"
            video.mkdir(exist_ok=True)
            for number, index in enumerate(range(0, len(elapsed), 5)):
                canvas = Image.new("RGB", (width * 3, width + 48), "white")
                draw = ImageDraw.Draw(canvas)
                for column, (name, row) in enumerate(zip(["Reference (evaluation only)", "Lowest observation-loss history", "Default history baseline"],
                    [truth[index], trajectories[best][index], trajectories[default][index]])):
                    fields = observer.capture(row)
                    canvas.paste(Image.fromarray((fields["rgb"] * 255).round().astype(np.uint8)), (column * width, 48))
                    draw.text((column * width + 5, 5), name, fill="black")
                    draw.text((column * width + 5, 21), f"t={elapsed[index]:.3f}s | gap={row['opening']*1000:.2f}mm", fill="black")
                canvas.save(video / f"{number:05d}.png")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i",
                str(video / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(root / "prediction-comparison.mp4")], check=True)
            shutil.copy2(video / "00000.png", root / "prediction-start.png")
            shutil.copy2(video / f"{number:05d}.png", root / "prediction-end.png")
    figure.savefig(root / "forecast-envelopes.png", dpi=150)
    plt.close(figure)
    stationary = [row for row in results if row["observation_protocol"] == "stationary"]
    informative = [row for row in results if row["observation_protocol"] == "fixed-small-impulse"]
    paired_noise_checks = []
    for first, second in zip(identifiers[::2], identifiers[1::2]):
        a = load_observations(root / "samples" / first / "public")
        b = load_observations(root / "samples" / second / "public")
        before_probe = min(action["time"] for action in b.public["observation_actions"])
        paired_noise_checks.append(all(np.array_equal(getattr(x, key), getattr(y, key))
            for x, y in zip(a.frames, b.frames) if x.elapsed <= before_probe
            for key in ("rgb", "depth", "object_alpha")))
    checks = {"all_selections_sealed": all(row["selection_seal_verified"] for row in results),
              "paired_preprobe_noisy_frames_identical": all(paired_noise_checks),
              "truth_not_memorized_in_bank": all(row["true_history_excluded_from_bank"] for row in results),
              "nonempty_observation_matches": all(row["accepted_count"] > 0 for row in results),
              "stationary_ambiguity_preserved": all(row["accepted_count"] == row["candidate_count"] for row in stationary),
              "informative_observation_reduces_candidates": all(row["accepted_count"] < row["candidate_count"] for row in informative)}
    report = {"status": "baseline_pipeline_checks_passed" if all(checks.values()) else "baseline_pipeline_checks_failed",
        "evaluation_code_sha256": digest(Path(__file__)),
        "run_directory": str(root), "checks": checks, "results": results,
        "method": "Finite physically reachable history replay and heuristic RGB-D/soft-mask filtering",
        "future_response_used_for_fitting": False, "new_inverse_baseline_implemented": True,
        "novel_method_implemented": False, "full_3d_fracture_validated": False,
        "limitations": ["One known interface, one-dimensional rigid translation, known rest pose/material/camera and observation actions.",
            "Four synthetic examples with fixed artificial noise; no real-sensor calibration or population-level generalization claim.",
            "Finite discrete hypothesis envelope is not a calibrated confidence interval; coverage is measured only after selection.",
            "Exact test histories excluded, but same law/asset and related synthetic generation remain favorable conditions.",
            "Known observation impulse is fixed manually, not a novel active-sensing method.",
            "Single-best tie breaking can select an arbitrary history under stationary ambiguity.",
            "No learned model, new physics or new state-estimation algorithm claimed."]}
    write_json(root / "report.json", report)
    artifacts = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts/cohesive-inverse"
    artifacts.mkdir(parents=True, exist_ok=True)
    for filename in ("report.json", "fit-settings.json", "forecast-envelopes.png", "prediction-comparison.mp4", "prediction-start.png", "prediction-end.png"):
        shutil.copy2(root / filename, artifacts / filename)
    print(json.dumps({"status": report["status"], "checks": checks, "results": results}, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


def child(root, stage, identifier=None):
    command = [sys.executable, "-u", str(Path(__file__)), "--run", str(root), "--stage", stage]
    if identifier:
        command += ["--sample", identifier]
    with (root / f"{stage}-{identifier or 'all'}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in process.stdout:
            log.write(line); tail = (tail + [line])[-20:]
            if line.startswith(("generated ", "fit ", "forecast ")):
                print(line, end="", flush=True)
        if process.wait():
            print("".join(tail), flush=True)
            raise subprocess.CalledProcessError(process.returncode, command)


def main():
    project = Path(os.environ["WM_PROJECT_ROOT"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=project / "config/cohesive-inverse.json")
    parser.add_argument("--stage", choices=["generate", "infer", "forecast", "evaluate"])
    parser.add_argument("--sample")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.stage:
        if args.stage == "generate": generate(args.run)
        elif args.stage == "infer": infer(args.run, args.sample)
        elif args.stage == "forecast": forecast(args.run, args.sample)
        else: evaluate(args.run)
        return
    if args.run:
        evaluate(args.run)
        return
    path = args.config if args.config.is_absolute() else project / args.config
    config = json.loads(path.read_text())
    root = Path(os.environ["WM_RESEARCH_ROOT"]) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-cohesive-inverse")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "private-experiment.json", config)
    write_json(root / "fit-settings.json", {"candidate_peaks": config["candidate_peaks"], "dt": config["inference_dt"], **config["acceptance"]})
    write_json(root / "code-sha256.json", {str(path.relative_to(project)): digest(path)
        for path in [Path(__file__), project / "scripts/cohesive_probe.py", *sorted((project / "wmfracture").glob("*.py"))]})
    print(f"OUTPUT={root}", flush=True)
    child(root, "generate")
    for identifier in json.loads((root / "dataset-index.json").read_text()):
        child(root, "infer", identifier)
        child(root, "forecast", identifier)
    child(root, "evaluate")


if __name__ == "__main__":
    main()
