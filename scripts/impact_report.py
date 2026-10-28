"""Compare the four small impact runs without treating separation as crack truth."""
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def main():
    artifacts = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts"
    reference = json.loads((Path(os.environ["WM_PROJECT_ROOT"]) / "config/first-impact.json").read_text())
    reference["physics"]["material"].pop("beta")
    reference["physics"]["initial_velocity"][2] = 0
    variants = []
    for directory in sorted((artifacts / "first-impact-variants").iterdir()):
        if not directory.is_dir():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        physics = manifest["config"]["physics"]
        normalized = json.loads(json.dumps(manifest["config"]))
        normalized["physics"]["material"].pop("beta")
        normalized["physics"]["initial_velocity"][2] = 0
        if normalized != reference:
            continue
        if physics["frames"] != 30:
            continue
        if summary["status"] != "completed" or not summary["finite_checks_passed"]:
            raise RuntimeError(f"Incomplete run: {directory}")
        if not summary["checks"]["precontact_gravity_passed"] or not summary["checks"]["floor_penetration_less_than_one_cell"]:
            raise RuntimeError(f"Mechanical sanity check failed: {directory}")
        run = Path(json.loads((directory / "run.json").read_text())["run_directory"])
        metrics = np.genfromtxt(directory / "metrics.csv", delimiter=",", names=True)
        variants.append({"beta": physics["material"]["beta"], "speed": -physics["initial_velocity"][2],
                         "directory": directory, "run": run, "summary": summary, "manifest": manifest, "metrics": metrics})
    variants.sort(key=lambda item: (-item["beta"], item["speed"]))
    if len(variants) != 4 or {(item["beta"], item["speed"]) for item in variants} != {(1.0, 6.0), (1.0, 10.0), (0.1, 6.0), (0.1, 10.0)}:
        raise RuntimeError("Run the four documented beta/speed combinations first")
    controlled_configs = set()
    for item in variants:
        config = json.loads(json.dumps(item["manifest"]["config"]))
        config["physics"]["material"].pop("beta")
        config["physics"]["initial_velocity"][2] = 0
        controlled_configs.add(json.dumps(config, sort_keys=True))
    if len(controlled_configs) != 1:
        raise RuntimeError("Configurations differ beyond beta and impact speed")
    fingerprints = {json.dumps(item["manifest"]["local_code_sha256"], sort_keys=True) for item in variants}
    if len(fingerprints) != 1:
        raise RuntimeError("Variant code fingerprints differ; rerun with the same implementation")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for item in variants:
        values = item["metrics"]
        label = f"beta={item['beta']:g}, speed={item['speed']:g}"
        for axis, column in zip(axes, ["com_z", "nonrigid_rms", "Jp_changed_fraction"]):
            axis.plot(values["time"], values[column], label=label)
    for axis, title in zip(axes, ["Center of mass height", "Shape change after rigid alignment", "Changed internal history fraction"]):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Simulation time (s)")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Normalized simulation units")
    axes[0].axhline(.2, color="gray", ls="--", lw=1, label="Floor")
    axes[2].legend(fontsize=8)
    fig.suptitle("Adapted forward baseline: response comparison, not validated fracture", fontsize=12)
    fig.savefig(artifacts / "impact-comparison.png", dpi=150)
    plt.close(fig)
    combined_frames = artifacts / "impact-comparison-frames"
    combined_frames.mkdir(exist_ok=True)
    for frame in range(31):
        montage = Image.new("RGB", (1024, 1024), "white")
        for index, item in enumerate(variants):
            with Image.open(item["run"] / "frames" / f"{frame:05d}.png") as image:
                image = image.copy()
            ImageDraw.Draw(image).text((12, 32), f"beta={item['beta']:g} | initial speed={item['speed']:g} (normalized)", fill=(25, 25, 25))
            montage.paste(image, ((index % 2) * 512, (index // 2) * 512))
        montage.save(combined_frames / f"{frame:05d}.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i",
                    str(combined_frames / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(artifacts / "impact-comparison.mp4")], check=True)
    rows = []
    for item in variants:
        summary = item["summary"]
        rows.append({"beta": item["beta"], "initial_speed": item["speed"], "run_directory": str(item["run"]),
                     "wall_seconds": summary["wall_seconds"], "global_gpu_peak_mib": summary["device_memory_used_peak_mib"],
                     "precontact_gravity_vz_max_error": summary["checks"]["precontact_gravity_vz_max_error"],
                     "final_nonrigid_rms": summary["last_frame"]["nonrigid_rms"],
                     "max_neighbor_stretch_gt2_fraction": summary["checks"]["max_neighbor_stretch_gt2_fraction"],
                     "max_history_changed_fraction": summary["checks"]["max_history_changed_fraction"]})
    report = {"all_mechanical_checks_passed": True, "variants": rows,
              "simulation_seconds": .3, "video_slowdown_factor": 1 / (.01 * 15),
              "notes": ["GPU memory includes all processes and is sampled, not an exclusive per-run peak.",
                        "Normalized demo material values are not calibrated physical measurements.",
                        "Neighbor stretching and Jp changes are diagnostics, not fracture ground truth.",
                        "Nearest-carrier appearance can introduce cracks/holes unrelated to physical fracture.",
                        "GPU atomic operations make reruns non-bitwise-deterministic; this is not repeatability validation."]}
    (artifacts / "impact-comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
