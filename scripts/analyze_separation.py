"""Check particle-space fragmentation and render it next to the appearance."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from wmfracture.rendering import build_renderer, floor_gaussians
from wmfracture.separation import (spatial_components, ComponentTracker,
                                   persistent_motion_candidate, simultaneous_consensus)


def main():
    artifacts = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--min-mass", type=float, default=.01)
    args = parser.parse_args()
    if args.stride < 1 or not 0 < args.min_mass < 1:
        raise ValueError("Invalid analysis stride/mass threshold")
    run = args.run or Path(json.loads((artifacts / "first-impact-latest.json").read_text())["run_directory"])
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Analyze completed runs only")
    config = manifest["config"]
    dx = config["physics"]["domain_size"] / config["physics"]["grid"]
    output = run / "separation"
    output.mkdir(exist_ok=True)
    frame_output = output / "frames"
    frame_output.mkdir(exist_ok=True)
    factors = [1., 1.5, 2.]
    rasterizer, camera = build_renderer(config["render"])
    floor = floor_gaussians(config["physics"]["floor"]["height"], config["physics"]["domain_size"])
    palette = np.array([[.12, .42, .82], [.92, .29, .12], [.15, .7, .3], [.7, .2, .7], [.9, .65, .1]], dtype=np.float32)
    rows = {str(factor): [] for factor in factors}
    with h5py.File(run / "trajectory.h5", "r") as trajectory:
        volumes = trajectory["volumes"][:]
        trackers = {str(factor): ComponentTracker(volumes) for factor in factors}
        frames = sorted(trajectory["frames"].keys())
        selected = list(range(0, len(frames), args.stride))
        if selected[-1] != len(frames) - 1:
            selected.append(len(frames) - 1)
        for output_index, frame_index in enumerate(selected):
            frame = trajectory["frames"][frames[frame_index]]
            x, v = frame["positions"][:], frame["velocities"][:]
            timestamp = float(frame.attrs["time"])
            primary = None
            primary_colors = None
            for factor in factors:
                labels, diagnostics = spatial_components(x, volumes, v, dx * factor, args.min_mass)
                track_ids = trackers[str(factor)].update(labels, diagnostics["major_components"], timestamp)
                persistent = [group for group in diagnostics["major_components"] if group["consecutive_duration"] >= .04 - 1e-8]
                diagnostics.update(frame=int(frames[frame_index]), time=timestamp,
                                   persistent_major_groups=len(persistent))
                # Motion criterion refers to the two largest groups; require both
                # to persist, rather than mixing them with unrelated dust tracks.
                diagnostics["separation_candidate"] = persistent_motion_candidate(
                    diagnostics["major_components"], diagnostics["two_largest_relative_speed"])
                rows[str(factor)].append(diagnostics)
                if factor == 1.5:
                    primary = diagnostics
                    colors = np.full((len(x), 3), .55, dtype=np.float32)
                    for identifier in np.unique(track_ids):
                        if identifier:
                            colors[track_ids == identifier] = palette[(identifier - 1) % len(palette)]
                    primary_colors = colors
                    np.savez_compressed(output / f"labels-{frame_index:05d}.npz", component=labels, track_id=track_ids)
            with torch.no_grad():
                points = torch.tensor(x, device="cuda")
                covariance = torch.tensor([(dx * .22) ** 2, 0, 0, (dx * .22) ** 2, 0, (dx * .22) ** 2], device="cuda").repeat(len(points), 1)
                all_points = torch.cat([points, floor[0]])
                image, _, _ = rasterizer(means3D=all_points, means2D=torch.zeros_like(all_points),
                    colors_precomp=torch.cat([torch.tensor(primary_colors, device="cuda"), floor[2]]),
                    opacities=torch.cat([torch.full((len(points), 1), .95, device="cuda"), floor[3]]),
                    cov3D_precomp=torch.cat([covariance, floor[1]]))
            particles = Image.fromarray((image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            ImageDraw.Draw(particles).text((12, 12), f"Particle groups | t={timestamp:.3f}s | radius=1.5dx", fill=(20, 20, 20))
            ImageDraw.Draw(particles).text((12, 32), f">={args.min_mass:.0%} mass groups: {len(primary['major_components'])}", fill=(20, 20, 20))
            with Image.open(run / "frames" / f"{frame_index:05d}.png") as appearance:
                combined = Image.new("RGB", (camera.width * 2, camera.height), "white")
                combined.paste(appearance, (0, 0))
                combined.paste(particles, (camera.width, 0))
            combined.save(frame_output / f"{output_index:05d}.png")
            print(f"frame={frame_index} t={timestamp:.3f} major@1.5dx={len(primary['major_components'])} "
                  f"largest_mass={primary['largest_mass_fraction']:.4f}", flush=True)
    sensitivity = {}
    for factor in factors:
        data = rows[str(factor)]
        initially_connected = data[0]["largest_mass_fraction"] >= .98
        sensitivity[str(factor)] = {"initial_largest_mass_fraction": data[0]["largest_mass_fraction"],
                                   "initially_connected": initially_connected,
                                   "max_major_groups": max(len(row["major_components"]) for row in data),
                                   "persistent_separation_candidate": initially_connected and any(row["separation_candidate"] for row in data)}
    consensus_frames = simultaneous_consensus(rows, {key: item["initially_connected"] for key, item in sensitivity.items()})
    project = Path(os.environ["WM_PROJECT_ROOT"])
    report = {"run_directory": str(run), "config": config, "dx": dx, "mass_threshold": args.min_mass,
              "min_persistent_duration": .04, "min_relative_speed": .1, "analysis_stride": args.stride,
              "radius_sensitivity": sensitivity, "frames": rows,
              "robust_particle_separation_candidate": bool(consensus_frames), "consensus_frames": consensus_frames,
              "analysis_code_sha256": {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in [Path(__file__), project / "wmfracture/separation.py", project / "wmfracture/rendering.py"]},
              "fracture_accuracy_validated": False,
              "limitations": ["Spatial disconnection is not proof of material fracture; touching fragments may remain connected.",
                              "Connectivity depends on sample density and radius; thresholds are normalized by MPM dx.",
                              "Mass fractions assume constant density, as in these experiments.",
                              f"Components below {args.min_mass:.1%} mass are reported as dust, not claimed as major fragments.",
                              "Particle-view colors are diagnostic track IDs, independent of Gaussian opacity and color."]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for factor in factors:
        data = rows[str(factor)]
        axes[0].plot([row["time"] for row in data], [row["largest_mass_fraction"] for row in data], label=f"radius={factor:g}dx")
        axes[1].plot([row["time"] for row in data], [len(row["major_components"]) for row in data], label=f"radius={factor:g}dx")
    for axis, title in zip(axes, ["Largest particle group mass fraction", "Groups with at least 1% mass"]):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Simulation time (s)")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.savefig(output / "connectivity.png", dpi=150)
    plt.close(fig)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10", "-i",
                    str(frame_output / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(output / "appearance-vs-particles.mp4")], check=True)
    preview = artifacts / "separation" / run.name
    preview.mkdir(parents=True, exist_ok=True)
    for filename in ("report.json", "connectivity.png", "appearance-vs-particles.mp4"):
        shutil.copy2(output / filename, preview / filename)
    shutil.copy2(frame_output / f"{len(selected) - 1:05d}.png", preview / "last-frame.png")
    (artifacts / "separation-latest.json").write_text(json.dumps({"directory": str(preview), "run_directory": str(run)}, indent=2))
    print(json.dumps({"output": str(preview), "sensitivity": sensitivity,
                      "robust_particle_separation_candidate": report["robust_particle_separation_candidate"]}, indent=2))


if __name__ == "__main__":
    main()
