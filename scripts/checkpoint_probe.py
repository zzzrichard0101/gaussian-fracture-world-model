"""Fresh-process restart validation, repeated rollout, and hidden-state ablation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import gc

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import warp as wp

from wmfracture.checkpoint import load_checkpoint
from wmfracture.rendering import SimulationRenderer


def launch(arguments, log_path):
    command = [sys.executable, "-u", *map(str, arguments)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    output, tail = None, []
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            tail.append(line)
            tail = tail[-15:]
            if line.startswith("OUTPUT="):
                output = Path(line.strip().split("=", 1)[1])
            if line.startswith(("OUTPUT=", "CHECKPOINT=", "frame 010", "frame 030", "resumed frame=10 ", "resumed frame=30 ")):
                print(line, end="", flush=True)
    code = process.wait()
    if code:
        print("".join(tail), flush=True)
        raise subprocess.CalledProcessError(code, command)
    if output is None:
        raise RuntimeError(f"Missing output path in {log_path}")
    return output


def rms(array):
    return float(np.sqrt(np.mean(np.asarray(array, dtype=np.float64) ** 2)))


def errors(reference, other, scales):
    return {
        "position_rms_dx": float(np.sqrt(np.mean(np.sum((reference["positions"].astype(np.float64) - other["positions"]) ** 2, axis=1)))) / scales["dx"],
        "velocity_rms_impact_speed": float(np.sqrt(np.mean(np.sum((reference["velocities"].astype(np.float64) - other["velocities"]) ** 2, axis=1)))) / scales["speed"],
        **{name + "_relative_rms": rms(reference[name].astype(np.float64) - other[name]) / scales[name]
           for name in ("Jp", "trial_F", "total_F")},
    }


def main():
    project = Path(os.environ["WM_PROJECT_ROOT"])
    scripts = project / "scripts"
    validation = json.loads((project / "config/checkpoint-validation.json").read_text())
    split, until = validation["split_frame"], validation["until_frame"]
    config = json.loads((project / "config/stateful-impact.json").read_text())
    if not 0 < split < until <= config["physics"]["frames"]:
        raise ValueError("Invalid checkpoint validation frame range")
    output = Path(os.environ["WM_RESEARCH_ROOT"]) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-checkpoint-probe")
    output.mkdir(parents=True, exist_ok=False)
    print(f"PROBE_OUTPUT={output}", flush=True)
    (output / "validation-config.json").write_text(json.dumps(validation, indent=2))
    reference = launch([scripts / "first_impact.py", "--config", project / "config/stateful-impact.json",
                        "--checkpoint-frame", split, "--checkpoint-frame", until], output / "reference.log")
    split_checkpoint = reference / "checkpoints" / f"frame-{split:05d}.h5"
    runs = {"continuous": reference}
    for name in ("resumed-a", "resumed-b", "history-reset"):
        print(f"Starting fresh process: {name}", flush=True)
        arguments = [scripts / "resume_impact.py", "--checkpoint", split_checkpoint,
                     "--until-frame", until, "--output", output / name]
        if name == "history-reset":
            arguments.append("--reset-hidden-state")
        runs[name] = launch(arguments, output / (name + ".log"))
    trajectories = {name: h5py.File(directory / "trajectory.h5", "r") for name, directory in runs.items()}
    records = {name: [] for name in runs if name != "continuous"}
    repeat_records = []
    dx = config["physics"]["domain_size"] / config["physics"]["grid"]
    speed = np.linalg.norm(config["physics"]["initial_velocity"])
    try:
        for frame in range(split, until + 1):
            values = {name: {field: stream[f"frames/{frame:05d}/{field}"][:] for field in ("positions", "velocities", "Jp", "trial_F", "total_F")}
                      for name, stream in trajectories.items()}
            baseline = values["continuous"]
            scales = {"dx": dx, "speed": max(speed, 1e-8),
                      **{field: max(rms(baseline[field]), 1e-8) for field in ("Jp", "trial_F", "total_F")}}
            timestamp = float(trajectories["continuous"][f"frames/{frame:05d}"].attrs["time"])
            for name in records:
                records[name].append({"frame": frame, "time": timestamp, **errors(baseline, values[name], scales)})
            repeat_records.append({"frame": frame, "time": timestamp, **errors(values["resumed-a"], values["resumed-b"], scales)})
    finally:
        for stream in trajectories.values():
            stream.close()
    # Load final snapshots independently to compare both images and otherwise
    # unrecorded hidden arrays (e.g. C and elastic F), without reading the PLY.
    images, final_hidden, audits = {}, {}, {}
    selected_fields = ("particle_F", "particle_C", "particle_stress", "particle_Jp", "particle_F_trial")
    for name, directory in runs.items():
        final_path = (directory / "checkpoints" / f"frame-{until:05d}.h5") if name == "continuous" else directory / "checkpoint-final.h5"
        loaded = load_checkpoint(final_path)
        images[name] = SimulationRenderer(loaded.runtime.config, loaded.appearance).image(loaded.runtime).permute(1, 2, 0).cpu().numpy()
        final_hidden[name] = {field: getattr(loaded.runtime.sim.mpm_state, field).numpy() for field in selected_fields}
        if name != "continuous":
            audits[name] = json.loads((directory / "restore-audit.json").read_text())
        del loaded
        gc.collect()
    hidden_errors = {name: {field: {"rms": rms(final_hidden[name][field] - final_hidden["continuous"][field]),
                                  "max_abs": float(np.abs(final_hidden[name][field] - final_hidden["continuous"][field]).max())}
                            for field in selected_fields} for name in records}
    comparisons = {}
    for name in records:
        maxima = {key: max(row[key] for row in records[name]) for key in validation["tolerances"] if key != "image_mae_255"}
        maxima["image_mae_255"] = float(np.abs(images[name] - images["continuous"]).mean() * 255)
        comparisons[name] = {"max_errors": maxima,
                             "within_declared_tolerances": all(maxima[key] <= value for key, value in validation["tolerances"].items()),
                             "per_frame": records[name]}
    repeat = {key: max(row[key] for row in repeat_records) for key in repeat_records[0] if key not in ("frame", "time")}
    repeat["image_mae_255"] = float(np.abs(images["resumed-a"] - images["resumed-b"]).mean() * 255)
    exact_restore = all(item["status"] == "restored-exact" and item["verified_arrays"] >= 38 for item in audits.values())
    initial_match = all(records[name][0][key] == 0 for name in ("resumed-a", "resumed-b") for key in repeat if key != "image_mae_255")
    passed = exact_restore and initial_match and all(comparisons[name]["within_declared_tolerances"] for name in ("resumed-a", "resumed-b"))
    report = {"status": "passed" if passed else "failed", "probe_directory": str(output),
              "reference_run": str(reference), "checkpoint": str(split_checkpoint), "validation": validation,
              "fresh_process_branches": {key: str(value) for key, value in runs.items()},
              "exact_restore_audits": audits, "initial_dynamic_arrays_match": initial_match,
              "comparisons_to_continuous": comparisons, "repeat_a_vs_b_max_errors": repeat,
              "final_hidden_state_errors": hidden_errors,
              "ablation_exceeds_full_restore_position_error": comparisons["history-reset"]["max_errors"]["position_rms_dx"] >
                    5 * max(comparisons["resumed-a"]["max_errors"]["position_rms_dx"], 1e-12),
              "scope": "Frame-boundary restart of static-floor/box APIC or guarded-FLIP pipeline. Fracture accuracy is not validated.",
              "limitations": ["Post-restart evolution is tolerance-checked, not claimed bitwise equal; GPU atomic operations vary.",
                              "Two independent resumes provide a small repeat baseline, not a confidence interval.",
                              "The ablation deliberately retains physical parameters/x/v while resetting F, C, Jp, stress, covariance and geometric history.",
                              "Moving/custom callbacks and external forces are rejected by this checkpoint schema."]}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for name, data in records.items():
        for axis, field in zip(axes, ("position_rms_dx", "Jp_relative_rms", "total_F_relative_rms")):
            axis.plot([row["time"] for row in data], [max(row[field], 1e-10) for row in data], label=name)
    for axis, field in zip(axes, ("position_rms_dx", "Jp_relative_rms", "total_F_relative_rms")):
        axis.set_title(field, fontsize=10)
        axis.set_xlabel("Simulation time (s)")
        axis.set_yscale("log")
        axis.grid(alpha=.25)
        axis.axhline(validation["tolerances"][field], color="gray", ls="--", lw=1)
    axes[-1].legend(fontsize=8)
    fig.savefig(output / "restart-errors.png", dpi=150)
    plt.close(fig)
    columns = [("continuous", "Continuous"), ("resumed-a", "Full restore"), ("history-reset", "History reset (ablation)")]
    frames = output / "comparison-frames"
    frames.mkdir()
    for index, frame in enumerate(range(split, until + 1)):
        montage = Image.new("RGB", (1536, 512), "white")
        for column, (name, label) in enumerate(columns):
            with Image.open(runs[name] / "frames" / f"{frame:05d}.png") as source:
                panel = source.copy()
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 512, 46), fill="white")
            draw.text((12, 12), label, fill=(20, 20, 20))
            draw.text((12, 30), f"frame={frame} | t={frame * config['physics']['frame_dt']:.3f}s", fill=(20, 20, 20))
            montage.paste(panel, (column * 512, 0))
        montage.save(frames / f"{index:05d}.png")
    shutil.copy2(frames / f"{until - split:05d}.png", output / "restart-comparison.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", "10", "-i", str(frames / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output / "restart-comparison.mp4")], check=True)
    preview = project / "artifacts/checkpoint-probe"
    preview.mkdir(parents=True, exist_ok=True)
    for filename in ("report.json", "restart-errors.png", "restart-comparison.png", "restart-comparison.mp4", "validation-config.json"):
        shutil.copy2(output / filename, preview / filename)
    print(json.dumps({"status": report["status"], "output": str(preview), "checkpoint": str(split_checkpoint),
                      "comparisons": {name: value["max_errors"] for name, value in comparisons.items()},
                      "repeat_baseline": repeat}, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
