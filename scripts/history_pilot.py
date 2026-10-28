"""Sequential physical prehistory -> RGB-D observation -> common re-impact pilot.

No inverse model is fitted here. Raw observations and privileged evaluation
states live in separate files. Child processes bound CUDA memory per history.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare_stage(root):
    import torch
    import warp as wp
    from first_impact import prepare
    from wmfracture.runtime import build_runtime
    from wmfracture.checkpoint import save_checkpoint

    wp.init()
    config = json.loads((root / "base-config.json").read_text())
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    with torch.no_grad():
        (initial, volumes, covariance, opacity, colors, binding, offsets, source_ids,
         chosen, asset_info, beta, reinforced) = prepare(config)
    runtime = build_runtime(config, initial, volumes, beta)
    appearance = dict(initial=initial, volumes=volumes, covariance=covariance, opacity=opacity, colors=colors,
                      binding=binding, offsets=offsets, source_ids=source_ids, chosen=chosen,
                      beta=torch.as_tensor(beta, device="cuda"), reinforced=torch.as_tensor(reinforced, device="cuda"))
    save_checkpoint(root / "initial.h5", runtime, appearance, {"asset": asset_info})
    write_json(root / "asset.json", asset_info)
    print(f"prepared particles={len(initial)} Gaussians={len(binding)}", flush=True)


def rollout_stage(root, name):
    import torch
    import warp as wp
    from first_impact import GPUMonitor
    from wmfracture.checkpoint import load_checkpoint, save_checkpoint
    from wmfracture.loading import validate_schedule
    from wmfracture.observations import GaussianRGBD
    from wmfracture.rendering import floor_gaussians
    from wmfracture.separation import (ComponentTracker, spatial_components,
                                       persistent_motion_candidate, simultaneous_consensus)

    output = root / name
    output.mkdir(exist_ok=False)
    write_json(output / "summary.json", {"status": "running"})
    pilot = json.loads((root / "pilot-config.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    case = next(item for item in cases if item["name"] == name)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with GPUMonitor() as monitor:
        loaded = load_checkpoint(root / "initial.h5")
        runtime, appearance = loaded.runtime, loaded.appearance
        runtime.config["physics"]["loading_schedule"] = validate_schedule(case["schedule"])
        config = runtime.config
        write_json(output / "config.json", config)
        renderer = GaussianRGBD(config["render"])
        camera = renderer.camera
        write_json(output / "camera.json", {"world_to_camera": camera.world_to_camera.tolist(),
            "fovx": camera.fovx, "fovy": camera.fovy, "width": camera.width, "height": camera.height,
            "depth_definition": "alpha-normalized Gaussian center camera-z, floor included",
            "object_mask_definition": "composited object_alpha > 0.5, occluded by floor"})
        floor = floor_gaussians(config["physics"]["floor"]["height"], config["physics"]["domain_size"])
        volumes = appearance["volumes"].cpu().numpy()
        dx = runtime.sim.mpm_model.dx
        factors = [1., 1.5, 2.]
        rows = {str(factor): [] for factor in factors}
        trackers = {str(factor): ComponentTracker(volumes) for factor in factors}
        frames_dir = output / "frames"
        frames_dir.mkdir()
        image_frames = []
        with h5py.File(output / "observations.h5", "x") as observations, h5py.File(output / "evaluation.h5", "x") as evaluation:
            observations.attrs["scope"] = "Observation interval only; no prehistory labels or internal state."
            observations.attrs["camera_json"] = json.dumps(json.loads((output / "camera.json").read_text()))
            evaluation.attrs["scope"] = "PRIVILEGED evaluation only: prehistory, future response, particle IDs and internal states."
            evaluation.create_dataset("volumes", data=volumes)
            for frame in range(pilot["frames"] + 1):
                if frame:
                    runtime.advance_frame()
                sim = runtime.sim
                positions = sim.export_particle_x_to_torch().cpu().numpy()
                velocities = sim.export_particle_v_to_torch().cpu().numpy()
                history = wp.to_torch(sim.mpm_state.particle_Jp).cpu().numpy()
                trial = wp.to_torch(sim.mpm_state.particle_F_trial)
                if (not all(np.isfinite(array).all() for array in (positions, velocities, history))
                    or not torch.isfinite(trial).all() or not torch.isfinite(runtime.total_F_tensor).all()
                    or bool(wp.to_torch(runtime.invalid_F).any())):
                    raise RuntimeError(f"Invalid physical/deformation state at frame {frame}")
                group = evaluation.create_group(f"frames/{frame:05d}")
                group.attrs["time"] = sim.time
                for key, array in [("positions", positions), ("velocities", velocities), ("Jp", history)]:
                    group.create_dataset(key, data=array, compression="gzip", compression_opts=1)
                if frame == max(pilot["observation_frames"]):
                    group.create_dataset("trial_F", data=trial.cpu().numpy(), compression="gzip", compression_opts=1)
                    group.create_dataset("total_F", data=runtime.total_F_tensor.cpu().numpy(), compression="gzip", compression_opts=1)
                    save_checkpoint(output / "observation-end.h5", runtime, appearance,
                                    {"scope": "privileged candidate state, not input observation", "case": case})
                if frame % pilot["render_stride"] == 0 or frame in pilot["observation_frames"] or frame == pilot["frames"]:
                    fields = renderer.capture(runtime, appearance, floor)
                    image_frames.append(frame)
                    Image.fromarray((fields["rgb"] * 255).round().astype(np.uint8)).save(frames_dir / f"{frame:05d}.png")
                    if frame in pilot["observation_frames"]:
                        target = observations.create_group(f"frames/{frame:05d}")
                        target.attrs["time"] = sim.time
                        for key, array in fields.items():
                            target.create_dataset(key, data=array, compression="gzip", compression_opts=1)
                if frame == 0 or (frame >= pilot["reimpact_frame"] and frame % 4 == 0):
                    for factor in factors:
                        labels, diagnostics = spatial_components(positions, volumes, velocities, dx * factor)
                        trackers[str(factor)].update(labels, diagnostics["major_components"], sim.time)
                        diagnostics.update(frame=frame, time=float(sim.time))
                        diagnostics["separation_candidate"] = persistent_motion_candidate(
                            diagnostics["major_components"], diagnostics["two_largest_relative_speed"])
                        rows[str(factor)].append(diagnostics)
                if frame in (0, max(pilot["observation_frames"]), pilot["reimpact_frame"] + 2, pilot["frames"]):
                    print(f"{name} frame={frame} t={sim.time:.3f} impulses={len(runtime.applied_loads)}", flush=True)
        initially_connected = {key: data[0]["largest_mass_fraction"] >= .98 for key, data in rows.items()}
        consensus = simultaneous_consensus(rows, initially_connected)
        write_json(output / "separation.json", {"radius_rows": rows, "initially_connected": initially_connected,
            "consensus_frames": consensus, "particle_separation_candidate": bool(consensus),
            "fracture_accuracy_validated": False,
            "note": "Sparse particle-space diagnostic; no grid/dt convergence or material crack validation."})
        summary = {"status": "completed", "case": case, "wall_seconds": time.perf_counter() - started,
                   "initial_verified_arrays": loaded.verified_arrays, "physics_particles": len(volumes),
                   "render_gaussians": len(appearance["binding"]), "image_frames": image_frames,
                   "applied_loads": runtime.applied_loads, "substeps": runtime.global_step,
                   "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                   "particle_separation_candidate": bool(consensus)}
    summary["device_memory_peak_mib"] = max(monitor.samples) if monitor.samples else None
    summary["device_memory_note"] = "Whole device includes other applications; sampled every 0.2s."
    write_json(output / "summary.json", summary)


def position_rms(first, second):
    difference = np.asarray(first, dtype=np.float64) - second
    return float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1))))


def compare_pair(root, first, second, pilot, dx):
    obs_rows = []
    with h5py.File(root / first / "observations.h5") as a, h5py.File(root / second / "observations.h5") as b:
        for frame in pilot["observation_frames"]:
            x, y = a[f"frames/{frame:05d}"], b[f"frames/{frame:05d}"]
            mask_a, mask_b = x["object_alpha"][:] > .5, y["object_alpha"][:] > .5
            union, intersection = mask_a | mask_b, mask_a & mask_b
            if not union.any() or not intersection.any():
                raise RuntimeError("Empty observation object/intersection mask")
            rgb = float(np.abs(x["rgb"][:] - y["rgb"][:])[union].mean() * 255)
            depth = float(np.sqrt(np.mean((x["depth"][:][intersection] - y["depth"][:][intersection]) ** 2))) / dx
            obs_rows.append({"frame": frame, "rgb_mae_255": rgb, "depth_rms_dx": depth,
                             "mask_iou": float(intersection.sum() / union.sum())})
    future, state_diagnostics = [], {}
    with h5py.File(root / first / "evaluation.h5") as a, h5py.File(root / second / "evaluation.h5") as b:
        for frame in range(pilot["reimpact_frame"] + 1, pilot["frames"] + 1):
            x, y = a[f"frames/{frame:05d}/positions"][:], b[f"frames/{frame:05d}/positions"][:]
            future.append({"frame": frame, "position_rms_dx": position_rms(x, y) / dx,
                           "translation_removed_rms_dx": position_rms(x - x.mean(0), y - y.mean(0)) / dx})
        frame = max(pilot["observation_frames"])
        x, y = a[f"frames/{frame:05d}"], b[f"frames/{frame:05d}"]
        for key in ("positions", "velocities", "Jp", "trial_F", "total_F"):
            state_diagnostics[key + "_rms"] = float(np.sqrt(np.mean((x[key][:].astype(np.float64) - y[key][:]) ** 2)))
    return {"first": first, "second": second, "observations": obs_rows,
            "rgb_mae_255_max": max(row["rgb_mae_255"] for row in obs_rows),
            "depth_rms_dx_max": max(row["depth_rms_dx"] for row in obs_rows),
            "mask_iou_min": min(row["mask_iou"] for row in obs_rows),
            "future_position_rms_dx_max": max(row["position_rms_dx"] for row in future),
            "future_translation_removed_rms_dx_max": max(row["translation_removed_rms_dx"] for row in future),
            "future": future, "privileged_observation_end_state_diagnostics": state_diagnostics}


def analyze(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pilot = json.loads((root / "pilot-config.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    config = json.loads((root / "base-config.json").read_text())
    dx = config["physics"]["domain_size"] / config["physics"]["grid"]
    primary = [item["name"] for item in cases if "repeat_of" not in item]
    repeats = [compare_pair(root, item["repeat_of"], item["name"], pilot, dx)
               for item in cases if "repeat_of" in item]
    repeat_floor = max(item["future_position_rms_dx_max"] for item in repeats)
    threshold = pilot["pilot_thresholds"]
    future_cutoff = max(threshold["future_position_rms_dx_min"], threshold["repeatability_multiplier"] * repeat_floor)
    pairs = [compare_pair(root, first, second, pilot, dx) for first, second in itertools.combinations(primary, 2)]
    for pair in pairs:
        pair["observation_score"] = max(pair["rgb_mae_255_max"] / threshold["rgb_mae_255_max"],
            pair["depth_rms_dx_max"] / threshold["depth_rms_dx_max"],
            (1 - pair["mask_iou_min"]) / (1 - threshold["mask_iou_min"]))
        pair["observation_similar"] = pair["observation_score"] <= 1
        pair["future_distinct"] = pair["future_position_rms_dx_max"] > future_cutoff
    eligible = [item for item in pairs if item["observation_similar"] and item["future_distinct"]]
    selected = max(eligible, key=lambda item: item["future_position_rms_dx_max"]) if eligible else min(pairs, key=lambda item: item["observation_score"])
    summaries = {item["name"]: json.loads((root / item["name"] / "summary.json").read_text()) for item in cases}
    report = {"status": "pilot_completed", "run_directory": str(root), "pilot_config": pilot,
              "gate_a": "candidate_found" if eligible else "not_demonstrated_in_this_sweep",
              "gate_b": "not_validated", "inverse_algorithm_implemented": False,
              "hidden_plastic_state_causality_isolated": False,
              "repeatability_max_position_rms_dx": repeat_floor, "future_distinct_cutoff_dx": future_cutoff,
              "eligible_pairs": len(eligible), "selected_pair": selected,
              "selected_pair_reason": "largest future difference among eligible pairs" if eligible else "closest observations; does not pass Gate A",
              "pairs": pairs, "repeats": repeats, "summaries": summaries,
              "limitations": ["Synthetic one-camera alpha-expected center depth; no calibrated RGB-D sensor model.",
                  "Eight values of one local impulse strength, one object, fixed material; no exhaustive identifiability result.",
                  "Two repeated histories estimate numerical variability, not confidence intervals.",
                  "Small observation error does not establish equal full x/v or isolate plastic memory causality.",
                  "Sparse particle separation is not physical fracture convergence validation.",
                  "No learned inverse model, method comparison, novelty or real-world prediction claim."]}
    write_json(root / "report.json", report)
    n = len(primary)
    matrices = [np.zeros((n, n)), np.zeros((n, n))]
    for pair in pairs:
        i, j = primary.index(pair["first"]), primary.index(pair["second"])
        for matrix, key in zip(matrices, ["observation_score", "future_position_rms_dx_max"]):
            matrix[i, j] = matrix[j, i] = pair[key]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for axis, matrix, title in zip(axes, matrices, ["Observation discrepancy (<=1 similar)", "Future particle RMS (grid cells)"]):
        artist = axis.imshow(matrix, cmap="viridis")
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(n)); axis.set_yticks(range(n))
        axis.set_xlabel("History index"); axis.set_ylabel("History index")
        figure.colorbar(artist, ax=axis, shrink=.8)
    figure.savefig(root / "pairwise.png", dpi=150)
    plt.close(figure)
    pair_frames = root / "comparison-frames"
    pair_frames.mkdir(exist_ok=True)
    names = [selected["first"], selected["second"]]
    indices = summaries[names[0]]["image_frames"]
    width = pilot["render_width"]
    for sequence, frame in enumerate(indices):
        combined = Image.new("RGB", (width * 2, width + 48), "white")
        draw = ImageDraw.Draw(combined)
        phase = "prehistory" if frame < min(pilot["observation_frames"]) else "observation" if frame <= max(pilot["observation_frames"]) else "future action / response"
        for column, name in enumerate(names):
            with Image.open(root / name / "frames" / f"{frame:05d}.png") as image:
                combined.paste(image, (column * width, 48))
            draw.text((column * width + 6, 4), f"{name} | frame {frame} | {phase}", fill="black")
        combined.save(pair_frames / f"{sequence:05d}.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10", "-i",
                    str(pair_frames / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(root / "comparison.mp4")], check=True)
    montage = Image.new("RGB", (width * 3, width * 2 + 48), "white")
    draw = ImageDraw.Draw(montage)
    for row, name in enumerate(names):
        draw.text((6, row * (width + 24) + 4), name, fill="black")
        for column, frame in enumerate(pilot["observation_frames"]):
            with Image.open(root / name / "frames" / f"{frame:05d}.png") as image:
                montage.paste(image, (column * width, row * (width + 24) + 24))
    montage.save(root / "observation-pair.png")
    artifacts = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts/history-pilot"
    artifacts.mkdir(parents=True, exist_ok=True)
    for filename in ("report.json", "pairwise.png", "comparison.mp4", "observation-pair.png", "pilot-config.json"):
        shutil.copy2(root / filename, artifacts / filename)
    print(json.dumps({key: report[key] for key in ("gate_a", "gate_b", "eligible_pairs", "repeatability_max_position_rms_dx")}, indent=2), flush=True)


def child(stage, root, name=None):
    arguments = [sys.executable, "-u", str(Path(__file__)), "--stage", stage, "--run", str(root)]
    if name:
        arguments += ["--case", name]
    log_path = root / (name or stage)
    log_path = log_path.with_suffix(".log")
    tail = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            tail = (tail + [line])[-20:]
            if " frame=" in line or line.startswith("prepared "):
                print(line, end="", flush=True)
        if process.wait():
            print("".join(tail), flush=True)
            raise subprocess.CalledProcessError(process.returncode, arguments)


def main():
    project = Path(os.environ["WM_PROJECT_ROOT"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=project / "config/history-pilot.json")
    parser.add_argument("--run", type=Path, help="Analyze an existing completed pilot; not a physics resume")
    parser.add_argument("--stage", choices=["prepare", "rollout"])
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_stage(args.run)
        return
    if args.stage == "rollout":
        rollout_stage(args.run, args.case)
        return
    if args.run:
        analyze(args.run)
        return
    config_path = args.config if args.config.is_absolute() else project / args.config
    pilot = json.loads(config_path.read_text())
    obs = pilot["observation_frames"]
    if not (obs and pilot["prior_load"]["frame"] < min(obs) <= max(obs) < pilot["reimpact_frame"] < pilot["frames"]
            and pilot["future_load"]["frame"] == pilot["reimpact_frame"]):
        raise ValueError("Require prehistory < observations < common future impulse < end")
    config = json.loads((project / pilot["base_config"]).read_text())
    if config["physics"].get("loading_schedule"):
        raise ValueError("Pilot requires a base configuration without scheduled loads")
    config["physics"]["frames"] = pilot["frames"]
    config["render"]["width"] = pilot["render_width"]
    root = Path(os.environ["WM_RESEARCH_ROOT"]) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-history-pilot")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "pilot-config.json", pilot)
    write_json(root / "base-config.json", config)
    write_json(root / "code-fingerprint.json", {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), *sorted((project / "wmfracture").glob("*.py"))]})
    cases = []
    for index, strength in enumerate(pilot["history_strengths"]):
        prior = copy.deepcopy(pilot["prior_load"])
        prior["delta_velocity"] = [float(strength * value) for value in prior["delta_velocity"]]
        cases.append({"name": f"history-{index:02d}", "strength": strength, "schedule": [prior, copy.deepcopy(pilot["future_load"])]})
    for index in pilot["repeat_indices"]:
        case = copy.deepcopy(cases[index])
        case["repeat_of"], case["name"] = case["name"], case["name"] + "-repeat"
        cases.append(case)
    write_json(root / "cases.json", cases)
    print(f"OUTPUT={root}", flush=True)
    child("prepare", root)
    for case in cases:
        child("rollout", root, case["name"])
    analyze(root)


if __name__ == "__main__":
    main()
