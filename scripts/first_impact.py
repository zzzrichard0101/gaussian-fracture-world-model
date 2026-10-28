"""Run a bounded first-impact baseline and retain provenance and trajectories."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import resource
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB

from wmfracture.geometry import (advect_gaussians, make_transform,
                                 rigid_alignment_rms, stratified_carriers, unpack_covariance)
from wmfracture.rendering import build_renderer, floor_gaussians
from wmfracture.diagnostics import mechanical_checks
from wmfracture.runtime import build_runtime
from wmfracture.checkpoint import save_checkpoint


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GPUMonitor:
    """Whole-device memory, including Warp and other desktop processes."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    text=True, capture_output=True, check=False)
            if result.returncode == 0:
                self.samples.append(int(result.stdout.strip()))
            self.stop.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)


def prepare(config):
    asset = Path(os.environ["WM_RESEARCH_ROOT"]) / "data/gaussianfluent/model" / config["asset"]
    ply = asset / "point_cloud/iteration_30000/point_cloud.ply"
    model = GaussianModel(3)
    model.load_ply(str(ply))
    pre = config["preprocessing"]
    valid = model.get_opacity[:, 0] > pre["opacity_threshold"]
    source_ids = valid.nonzero().squeeze(1)
    positions_world = model.get_xyz[valid].detach()
    covariance_world = unpack_covariance(model.get_covariance()[valid].detach())
    opacity = model.get_opacity[valid].detach().contiguous()
    # DC-only appearance avoids pretending unrotated SH directions are valid
    # after arbitrary rotation/deformation. This choice is recorded in manifest.
    colors = SH2RGB(model.get_features[valid, 0].detach()).clamp(0, 1).contiguous()
    rotation, scale, origin, center = make_transform(positions_world, pre["rotation_xyz_degrees"],
                                                    pre["object_extent"], pre["simulation_center"])
    positions = ((positions_world @ rotation.T - origin) * scale + center).contiguous()
    covariance = (scale ** 2 * (rotation @ covariance_world @ rotation.T)).contiguous()
    physics = config["physics"]
    chosen, volumes, cells = stratified_carriers(positions.cpu().numpy(), physics["grid"], physics["domain_size"],
                                               physics["particles_per_occupied_cell"], config["seed"])
    if len(chosen) > physics["max_particles"]:
        raise ValueError(f"{len(chosen)} carriers exceed max_particles; reduce grid/per-cell samples")
    chosen = torch.tensor(chosen, device="cuda", dtype=torch.long)
    initial = positions[chosen].contiguous()
    tree = cKDTree(initial.cpu().numpy())
    distances, binding_np = tree.query(positions.cpu().numpy(), workers=8)
    binding = torch.tensor(binding_np, device="cuda", dtype=torch.long)
    rest_offsets = (positions - initial[binding]).contiguous()
    physical_volumes = torch.tensor(volumes, device="cuda")
    heterogeneity = physics.get("seed_reinforcement", {"enabled": False})
    reinforced = np.zeros(len(initial), dtype=bool)
    beta = np.full(len(initial), physics["material"]["beta"], dtype=np.float32)
    if heterogeneity["enabled"]:
        dc = model.get_features[valid, 0].detach()
        dark = ((dc > heterogeneity["dc_min"]) & (dc < heterogeneity["dc_max"])).all(dim=1)
        if dark.any():
            seed_tree = cKDTree(positions[dark].cpu().numpy())
            seed_distance, _ = seed_tree.query(initial.cpu().numpy(), workers=8)
            reinforced = seed_distance <= heterogeneity["radius_world"] * scale
            beta[reinforced] = heterogeneity["beta"]
    manifest = {"source_ply": str(ply), "source_ply_sha256": sha256_file(ply),
                "source_gaussians": int(len(model.get_xyz)), "render_gaussians": len(positions),
                "physics_particles": len(initial), "occupied_cells": cells, "occupied_volume": float(volumes.sum()),
                "initial_bounds": [positions.amin(0).tolist(), positions.amax(0).tolist()],
                "binding_distance_max": float(distances.max()), "binding_distance_mean": float(distances.mean()),
                "appearance": "DC RGB; fixed opacity; no Jp-based opacity removal",
                "transfer": "Nearest carrier local affine offset/covariance using " + config["render"].get("deformation", "trial") + " deformation",
                "volume_rule": "Occupied Gaussian-center cells only; each cell contributes dx^3; no watertight filling",
                "seed_reinforcement": {"enabled": heterogeneity["enabled"], "particles": int(reinforced.sum()),
                                       "volume_fraction": float(volumes[reinforced].sum() / volumes.sum()),
                                       "note": "Upstream DC-color heuristic; not semantic seed ground truth."},
                "transform": {"rotation": rotation.cpu().tolist(), "scale": scale, "rotated_bbox_center": origin.cpu().tolist(),
                              "simulation_center": center.cpu().tolist()}}
    del model, positions_world, covariance_world
    return initial, physical_volumes, covariance, opacity, colors, binding, rest_offsets, source_ids, chosen, manifest, beta, reinforced


def run(config, output, checkpoint_frames=()):
    physics, display = config["physics"], config["render"]
    if physics["transfer"] not in ("apic", "guarded-flip"):
        raise ValueError("Supported transfers: apic, guarded-flip")
    if display["history_opacity_culling"]:
        raise ValueError("History opacity culling is intentionally unsupported in this baseline")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    wp.init()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        initial, volumes, covariance, opacity, colors, binding, offsets, source_ids, chosen, asset_info, beta, reinforced = prepare(config)
    print(json.dumps(asset_info, indent=2), flush=True)
    source = Path(os.environ["WM_SOURCE_ROOT"])
    manifest = {"status": "running", "config": config, "asset": asset_info,
                "upstream_commit": git_revision(source), "gaussian_splatting_commit": git_revision(source / "gaussian-splatting"),
                "python_torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "warp": importlib.metadata.version("warp-lang"),
                "device": torch.cuda.get_device_name(0), "started_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoints": [],
                "scope": "Small adapted forward baseline; optional frame-boundary checkpoints; fracture accuracy unvalidated."}
    project = Path(os.environ["WM_PROJECT_ROOT"])
    manifest["local_code_sha256"] = {str(path.relative_to(project)): sha256_file(path)
                                     for path in [Path(__file__), *sorted((project / "wmfracture").glob("*.py"))]}
    manifest["upstream_tracked_changes"] = subprocess.check_output(["git", "-C", str(source), "diff", "--stat"], text=True)
    manifest["asset_release"] = json.loads((project / "config/assets.json").read_text())
    deformation_mode = display.get("deformation", "trial")
    if deformation_mode not in ("trial", "total"):
        raise ValueError("Deformation must be trial or total")
    manifest["render_deformation"] = deformation_mode
    write_json(output / "manifest.json", manifest)
    runtime = build_runtime(config, initial, volumes, beta)
    sim = runtime.sim
    floor = physics["floor"]
    total_F_tensor, invalid_F = runtime.total_F_tensor, runtime.invalid_F
    appearance = dict(initial=initial, volumes=volumes, covariance=covariance, opacity=opacity, colors=colors,
                      binding=binding, offsets=offsets, source_ids=source_ids, chosen=chosen, beta=beta, reinforced=reinforced)
    rasterizer, camera = build_renderer(display)
    floor_pos, floor_cov, floor_colors, floor_opacity = floor_gaussians(floor["height"], physics["domain_size"])
    render_colors = torch.cat([colors, floor_colors])
    render_opacity = torch.cat([opacity, floor_opacity])
    write_json(output / "camera.json", {"world_to_camera": camera.world_to_camera.tolist(), "fovx": camera.fovx,
                                      "fovy": camera.fovy, "width": camera.width, "height": camera.height})
    frames_dir = output / "frames"
    frames_dir.mkdir()
    # Fixed initial neighbors diagnose separation; this is not fracture ground truth.
    _, neighbors = cKDTree(initial.cpu().numpy()).query(initial.cpu().numpy(), k=2)
    neighbor = torch.tensor(neighbors[:, 1], device="cuda")
    initial_edge_length = (initial - initial[neighbor]).norm(dim=1).clamp_min(1e-8)
    material = physics["material"]
    metrics = []
    write_json(output / "summary.json", {"status": "running", "scope": manifest["scope"]})
    with h5py.File(output / "trajectory.h5", "w") as trajectory:
        trajectory.attrs["resumable_checkpoint"] = False
        trajectory.attrs["has_external_checkpoints"] = bool(checkpoint_frames)
        trajectory.attrs["render_deformation"] = deformation_mode
        trajectory.create_dataset("initial_positions", data=initial.cpu().numpy())
        trajectory.create_dataset("volumes", data=volumes.cpu().numpy())
        trajectory.create_dataset("beta", data=beta)
        trajectory.create_dataset("reinforced", data=reinforced)
        trajectory.create_dataset("source_gaussian_ids", data=source_ids.cpu().numpy(), compression="gzip")
        trajectory.create_dataset("carrier_selected_indices", data=chosen.cpu().numpy())
        trajectory.create_dataset("gaussian_to_carrier", data=binding.cpu().numpy(), compression="gzip")
        global_step = 0
        for frame in range(physics["frames"] + 1):
            frame_start = time.perf_counter()
            substeps, step_dt = 0, 0.0
            if frame:
                substeps, step_dt = runtime.advance_frame()
                global_step = runtime.global_step
            positions = sim.export_particle_x_to_torch()
            velocities = sim.export_particle_v_to_torch()
            F_trial = wp.to_torch(sim.mpm_state.particle_F_trial)
            history = wp.to_torch(sim.mpm_state.particle_Jp)
            deformation = F_trial if deformation_mode == "trial" else total_F_tensor
            if deformation_mode == "total" and (wp.to_torch(invalid_F).any() or not torch.isfinite(deformation).all()):
                raise RuntimeError(f"Invalid accumulated deformation at frame {frame}")
            for name, value in [("x", positions), ("v", velocities), ("F_trial", F_trial), ("Jp", history),
                                ("C", wp.to_torch(sim.mpm_state.particle_C)), ("stress", wp.to_torch(sim.mpm_state.particle_stress))]:
                if not torch.isfinite(value).all():
                    bad = (~torch.isfinite(value)).reshape(len(initial), -1).any(dim=1).nonzero().squeeze(1)
                    sample = bad[:128]
                    np.savez_compressed(output / "failure-state.npz", field=name, frame=frame,
                        invalid_particle_count=len(bad), particle_ids=sample.cpu().numpy(),
                        positions=positions[sample].cpu().numpy(), velocities=velocities[sample].cpu().numpy(),
                        Jp=history[sample].cpu().numpy(), beta=beta[sample.cpu().numpy()])
                    raise RuntimeError(f"Non-finite {name} at frame {frame}")
            # F_trial is identity before the first step. Do not read the initially-zero elastic F.
            with torch.no_grad():
                render_positions, render_cov = advect_gaussians(positions, deformation, binding, offsets, covariance)
                render_positions = torch.cat([render_positions, floor_pos])
                render_cov = torch.cat([render_cov, floor_cov])
                rendered, radii, _ = rasterizer(means3D=render_positions, means2D=torch.zeros_like(render_positions),
                                               colors_precomp=render_colors, opacities=render_opacity, cov3D_precomp=render_cov)
            if not torch.isfinite(rendered).all() or not (radii > 0).any():
                raise RuntimeError(f"Invalid/empty render at frame {frame}")
            pixels = (rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(pixels)
            ImageDraw.Draw(image).text((12, 12), f"t={sim.time:.3f}s | adapted forward baseline", fill=(30, 30, 30))
            image.save(frames_dir / f"{frame:05d}.png")
            group = trajectory.create_group(f"frames/{frame:05d}")
            group.attrs["time"] = sim.time
            for name, value in [("positions", positions), ("velocities", velocities), ("Jp", history)]:
                group.create_dataset(name, data=value.cpu().numpy(), compression="gzip")
            group.create_dataset("trial_F", data=F_trial.cpu().numpy(), compression="gzip")
            if deformation_mode == "total":
                group.create_dataset("total_F", data=total_F_tensor.cpu().numpy(), compression="gzip")
            separation = (positions - positions[neighbor]).norm(dim=1) / initial_edge_length
            row = {"frame": frame, "time": sim.time, "substeps": substeps, "substep_dt": step_dt,
                   "com_z": float((positions[:, 2] * volumes).sum() / volumes.sum()), "min_z": float(positions[:, 2].min()),
                   "mean_vz": float(velocities[:, 2].mean()), "max_speed": float(velocities.norm(dim=1).max()),
                   "Jp_min": float(history.min()), "Jp_max": float(history.max()),
                   "Jp_changed_fraction": float(((history - material["alpha_0"]).abs() > 1e-4).float().mean()),
                   "nonrigid_rms": float(rigid_alignment_rms(initial, positions)),
                   "neighbor_stretch_gt2_fraction": float((separation > 2).float().mean()),
                   "det_F_trial_min": float(torch.linalg.det(F_trial).min()),
                   "kinetic_energy": float((0.5 * material["density"] * volumes * velocities.square().sum(dim=1)).sum()),
                   "visible_gaussians": int((radii[:len(binding)] > 0).sum()), "frame_wall_seconds": time.perf_counter() - frame_start}
            metrics.append(row)
            print(f"frame {frame:03d} t={sim.time:.3f} zmin={row['min_z']:.3f} maxv={row['max_speed']:.2f} "
                  f"history_changed={row['Jp_changed_fraction']:.3f} wall={row['frame_wall_seconds']:.2f}s", flush=True)
            with (output / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(metrics)
            if frame in checkpoint_frames:
                checkpoint_path = output / "checkpoints" / f"frame-{frame:05d}.h5"
                save_checkpoint(checkpoint_path, runtime, appearance,
                                provenance={"asset": asset_info, "run_manifest": manifest, "source_run": str(output)})
                manifest["checkpoints"].append(str(checkpoint_path))
                write_json(output / "manifest.json", manifest)
                print(f"CHECKPOINT={checkpoint_path}", flush=True)
    video = output / "first-impact.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", str(display["fps"]),
                    "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    selected_frames = [0, len(metrics) // 3, 2 * len(metrics) // 3, len(metrics) - 1]
    montage = Image.new("RGB", (camera.width * 4, camera.height), "white")
    for index, frame in enumerate(selected_frames):
        with Image.open(frames_dir / f"{frame:05d}.png") as image:
            montage.paste(image, (index * camera.width, 0))
    montage.save(output / "contact-sheet.png")
    summary = {"status": "completed", "finite_checks_passed": True, "simulation_seconds": sim.time,
               "substeps": global_step, "wall_seconds": time.perf_counter() - start,
               "physics_particles": len(initial), "render_gaussians": len(binding),
               "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
               "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "video_slowdown_factor": 1 / (display["fps"] * physics["frame_dt"]),
               "last_frame": metrics[-1], "scope": manifest["scope"]}
    summary["render_deformation"] = deformation_mode
    summary["checkpoints"] = manifest["checkpoints"]
    if deformation_mode == "total":
        summary["total_F_norm_max"] = float(total_F_tensor.norm(dim=(1, 2)).max())
    summary["checks"] = mechanical_checks(metrics, physics["initial_velocity"][2], material["g"][2],
                                           floor["height"], sim.mpm_model.dx)
    manifest["status"] = "completed"
    write_json(output / "manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(os.environ["WM_PROJECT_ROOT"]) / "config/first-impact.json")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--impact-speed", type=float)
    parser.add_argument("--transfer", choices=["apic", "guarded-flip"])
    parser.add_argument("--grid", type=int)
    parser.add_argument("--particles-per-cell", type=int)
    parser.add_argument("--reinforce-seeds", action="store_true")
    parser.add_argument("--seed-beta", type=float)
    parser.add_argument("--deformation", choices=["trial", "total"])
    parser.add_argument("--checkpoint-frame", type=int, action="append", default=[])
    args = parser.parse_args()
    if not args.config.is_absolute():
        args.config = Path(os.environ["WM_PROJECT_ROOT"]) / args.config
    config = json.loads(args.config.read_text())
    if args.deformation:
        config["render"]["deformation"] = args.deformation
    if args.transfer:
        config["physics"]["transfer"] = args.transfer
    if args.grid is not None:
        if args.grid < 16:
            raise ValueError("Grid must be at least 16")
        config["physics"]["grid"] = args.grid
    if args.particles_per_cell is not None:
        if args.particles_per_cell < 1:
            raise ValueError("Particles per cell must be positive")
        config["physics"]["particles_per_occupied_cell"] = args.particles_per_cell
    if args.reinforce_seeds:
        config["physics"]["seed_reinforcement"] = {"enabled": True, "dc_min": -2.5, "dc_max": -.5,
                                                   "radius_world": .03, "beta": 3000000000.0 if args.seed_beta is None else args.seed_beta}
    if args.seed_beta is not None and (not args.reinforce_seeds or not np.isfinite(args.seed_beta) or args.seed_beta < 0):
        raise ValueError("--seed-beta requires --reinforce-seeds and a finite nonnegative value")
    if args.frames is not None:
        config["physics"]["frames"] = args.frames
    if args.beta is not None:
        if not np.isfinite(args.beta) or args.beta < 0:
            raise ValueError("Beta must be nonnegative")
        config["physics"]["material"]["beta"] = args.beta
    if args.impact_speed is not None:
        if not np.isfinite(args.impact_speed) or args.impact_speed <= 0:
            raise ValueError("Impact speed must be positive")
        config["physics"]["initial_velocity"][2] = -args.impact_speed
    if config["physics"]["frames"] < 1:
        raise ValueError("At least one simulation frame is required")
    if any(frame < 0 or frame > config["physics"]["frames"] for frame in args.checkpoint_frame):
        raise ValueError("Checkpoint frames must fall inside this run")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + config_hash
    output = Path(os.environ["WM_RESEARCH_ROOT"]) / "runs" / name
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    print(f"OUTPUT={output}", flush=True)
    with GPUMonitor() as monitor:
        try:
            summary = run(config, output, checkpoint_frames=set(args.checkpoint_frame))
        except Exception as error:
            write_json(output / "summary.json", {"status": "failed", "error": repr(error)})
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                failed_manifest = json.loads(manifest_path.read_text())
                failed_manifest.update(status="failed", error=repr(error))
                write_json(manifest_path, failed_manifest)
            raise
    summary["device_memory_used_peak_mib"] = max(monitor.samples) if monitor.samples else None
    summary["device_memory_used_first_sample_mib"] = monitor.samples[0] if monitor.samples else None
    summary["device_memory_note"] = "Whole device sampled every ~0.2s; includes other processes; may miss short peaks."
    write_json(output / "summary.json", summary)
    artifacts = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts"
    artifacts.mkdir(exist_ok=True)
    variant_output = artifacts / "first-impact-variants" / config_hash
    variant_output.mkdir(parents=True, exist_ok=True)
    for filename in ("first-impact.mp4", "contact-sheet.png", "summary.json", "manifest.json", "metrics.csv"):
        shutil.copy2(output / filename, artifacts / ("first-impact-" + filename if filename != "first-impact.mp4" else filename))
        shutil.copy2(output / filename, variant_output / filename)
    write_json(variant_output / "run.json", {"run_directory": str(output)})
    write_json(artifacts / "first-impact-latest.json", {"run_directory": str(output), "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
