"""Replay different appearance transfers from the exact same recorded physics."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from first_impact import prepare
from wmfracture.geometry import advect_gaussians
from wmfracture.rendering import build_renderer, floor_gaussians


def main():
    artifacts = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("Stride must be positive")
    run = args.run or Path(json.loads((artifacts / "first-impact-latest.json").read_text())["run_directory"])
    config = json.loads((run / "config.json").read_text())
    with torch.no_grad():
        initial, volumes, cov, opacity, colors, binding, offsets, source_ids, chosen, info, _, _ = prepare(config)
    renderer, camera = build_renderer(config["render"])
    floor = floor_gaussians(config["physics"]["floor"]["height"], config["physics"]["domain_size"])
    full_colors, full_opacity = torch.cat([colors, floor[2]]), torch.cat([opacity, floor[3]])
    dx = config["physics"]["domain_size"] / config["physics"]["grid"]
    particle_cov = torch.tensor([(dx * .22) ** 2, 0, 0, (dx * .22) ** 2, 0, (dx * .22) ** 2], device="cuda").repeat(len(initial), 1)
    particle_colors = torch.tensor([.12, .42, .82], device="cuda").repeat(len(initial), 1)
    particle_opacity = torch.full((len(initial), 1), .95, device="cuda")
    output = run / "transfer-replay"
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    statistics = []
    with h5py.File(run / "trajectory.h5", "r") as trajectory:
        np.testing.assert_array_equal(trajectory["carrier_selected_indices"][:], chosen.cpu().numpy())
        np.testing.assert_array_equal(trajectory["gaussian_to_carrier"][:], binding.cpu().numpy())
        np.testing.assert_allclose(trajectory["initial_positions"][:], initial.cpu().numpy(), atol=0, rtol=0)
        frames = sorted(trajectory["frames"].keys())
        selected = list(range(0, len(frames), args.stride))
        if selected[-1] != len(frames) - 1:
            selected.append(len(frames) - 1)
        for index, frame_index in enumerate(selected):
            frame = trajectory["frames"][frames[frame_index]]
            if "total_F" not in frame or "trial_F" not in frame:
                raise ValueError("Replay needs a new run with --deformation total and recorded trial_F")
            positions = torch.tensor(frame["positions"][:], device="cuda")
            trial = torch.tensor(frame["trial_F"][:], device="cuda")
            total = torch.tensor(frame["total_F"][:], device="cuda")
            images = []
            with torch.no_grad():
                for deformation in (trial, total):
                    points, covariance = advect_gaussians(positions, deformation, binding, offsets, cov)
                    points = torch.cat([points, floor[0]])
                    image, _, _ = renderer(means3D=points, means2D=torch.zeros_like(points),
                        colors_precomp=full_colors, opacities=full_opacity, cov3D_precomp=torch.cat([covariance, floor[1]]))
                    images.append(image.clamp(0, 1))
                points = torch.cat([positions, floor[0]])
                image, _, _ = renderer(means3D=points, means2D=torch.zeros_like(points),
                    colors_precomp=torch.cat([particle_colors, floor[2]]),
                    opacities=torch.cat([particle_opacity, floor[3]]), cov3D_precomp=torch.cat([particle_cov, floor[1]]))
                images.append(image.clamp(0, 1))
            if not all(torch.isfinite(image).all() for image in images):
                raise RuntimeError("Non-finite replay image")
            difference = float((images[0] - images[1]).abs().mean())
            if frame_index == 0 and difference > 1e-6:
                raise RuntimeError("Initial identity-deformation renders must match")
            timestamp = float(frame.attrs["time"])
            montage = Image.new("RGB", (camera.width * 3, camera.height), "white")
            for column, (image, label) in enumerate(zip(images, ["Elastic/trial F (legacy)", "Accumulated kinematic F", "Physical particle centers"])):
                pixels = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                panel = Image.fromarray(pixels)
                draw = ImageDraw.Draw(panel)
                draw.text((12, 12), label, fill=(20, 20, 20))
                draw.text((12, 32), f"Same recorded physics | t={timestamp:.3f}s", fill=(20, 20, 20))
                montage.paste(panel, (column * camera.width, 0))
            montage.save(frames_dir / f"{index:05d}.png")
            statistics.append({"frame": frame_index, "time": timestamp, "pixel_mean_abs_difference": difference,
                               "mean_affine_difference": float((trial - total).square().sum(dim=(1, 2)).sqrt().mean())})
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10", "-i",
                    str(frames_dir / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output / "transfer-replay.mp4")], check=True)
    report = {"run_directory": str(run), "same_recorded_physics": True, "binding_reconstruction_verified": True,
              "initial_identity_render_match": True, "frames": statistics,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "note": "Transfer differences are isolated from physics; image difference is not a quality or fracture metric."}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    preview = artifacts / "transfer-replay" / run.name
    preview.mkdir(parents=True, exist_ok=True)
    for filename in ("transfer-replay.mp4", "report.json"):
        shutil.copy2(output / filename, preview / filename)
    shutil.copy2(frames_dir / f"{len(selected)-1:05d}.png", preview / "last-frame.png")
    (artifacts / "transfer-replay-latest.json").write_text(json.dumps({"directory": str(preview), "run_directory": str(run)}, indent=2))
    print(json.dumps({"output": str(preview), "same_recorded_physics": True, "initial_identity_render_match": True,
                      "last_frame": statistics[-1]}, indent=2))


if __name__ == "__main__":
    main()
