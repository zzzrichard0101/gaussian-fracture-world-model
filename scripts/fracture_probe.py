"""Bounded source-vs-adaptation forward audit; failures are retained as results."""
import argparse
import copy
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
from PIL import Image, ImageDraw


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) else copy.deepcopy(value)
    return result


def geometry_audit(points, grid, size, occupied_volume):
    from scipy.spatial import ConvexHull
    dx = size / grid
    low, high = points.min(0), points.max(0)
    cells = np.unique(np.floor(points / dx).astype(np.int32), axis=0)
    centers = (cells + .5) * dx
    center = (low + high) * .5
    inner_low, inner_high = center - .25 * (high - low), center + .25 * (high - low)
    actual = int(((centers >= inner_low) & (centers <= inner_high)).all(1).sum())
    axes = [np.arange(grid)[((np.arange(grid) + .5) * dx >= inner_low[i]) &
                           ((np.arange(grid) + .5) * dx <= inner_high[i])] for i in range(3)]
    possible = int(np.prod([len(axis) for axis in axes]))
    hull = float(ConvexHull(points).volume)
    return {"occupied_volume": occupied_volume, "particle_center_convex_hull_volume": hull,
            "occupied_cell_volume_over_hull": occupied_volume / hull,
            "central_half_bbox_occupied_cells": actual, "central_half_bbox_possible_cells": possible,
            "central_half_bbox_occupancy_fraction": actual / possible,
            "note": "Occupancy/convex hull are coarse geometry diagnostics, not watertightness or density ground truth."}


def install_failure_audit(sim, output):
    """Observe unchanged upstream stepping; capture inputs to the first bad step."""
    import torch
    import warp as wp
    original = sim.p2g2p

    def audited(step, dt, **kwargs):
        state = sim.mpm_state
        before_trial = wp.to_torch(state.particle_F_trial).clone()
        before_jp = wp.to_torch(state.particle_Jp).clone()
        original(step, dt, **kwargs)
        invalid = torch.zeros(sim.n_particles, dtype=torch.bool, device="cuda")
        fields = {name: wp.to_torch(getattr(state, name)) for name in
                  ("particle_x", "particle_v", "particle_F", "particle_F_trial", "particle_Jp", "particle_stress")}
        bad_counts = {}
        for name, value in fields.items():
            bad = ~torch.isfinite(value).reshape(sim.n_particles, -1).all(1)
            count = int(bad.sum())
            if count:
                bad_counts[name] = count
                invalid |= bad
        if invalid.any():
            identifiers = invalid.nonzero().flatten()[:64]
            payload = {"particle_ids": identifiers.cpu().numpy(), "before_trial_F": before_trial[identifiers].cpu().numpy(),
                       "before_Jp": before_jp[identifiers].cpu().numpy()}
            payload.update({name: value[identifiers].cpu().numpy() for name, value in fields.items()})
            payload.update({name: wp.to_torch(getattr(sim.mpm_model, name))[identifiers].cpu().numpy()
                            for name in ("mu", "kappa", "beta")})
            np.savez_compressed(output / "first-nonfinite-step.npz", **payload)
            write_json(output / "first-nonfinite-step.json", {"step": step, "time": float(sim.time), "dt": dt,
                "invalid_fields": bad_counts, "before_trial_finite": bool(torch.isfinite(before_trial).all()),
                "before_Jp_finite": bool(torch.isfinite(before_jp).all()), "M": float(sim.mpm_model.M),
                "xi": float(sim.mpm_model.xi), "hardening": float(sim.mpm_model.hardening)})
            raise RuntimeError(f"First non-finite step captured: {bad_counts}")
    sim.p2g2p = audited


def run_case(root, identifier):
    import torch
    import warp as wp
    from first_impact import prepare, GPUMonitor
    from wmfracture.runtime import build_runtime
    from wmfracture.rendering import SimulationRenderer
    from wmfracture.separation import (ComponentTracker, spatial_components,
                                      persistent_motion_candidate, simultaneous_consensus)

    specification = json.loads((root / "probe-config.json").read_text())
    case = next(item for item in specification["cases"] if item["id"] == identifier)
    config = merge(json.loads((root / "base-config.json").read_text()), case["overrides"])
    output = root / identifier
    output.mkdir(exist_ok=False)
    (output / "frames").mkdir()
    write_json(output / "config.json", config)
    wp.init()
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    report = {"status": "running", "case": case, "last_complete_frame": -1}
    write_json(output / "summary.json", report)
    with GPUMonitor() as monitor:
        try:
            with torch.no_grad():
                (initial, volumes, covariance, opacity, colors, binding, offsets, source_ids,
                 chosen, asset_info, beta, reinforced) = prepare(config)
            write_json(output / "asset.json", asset_info)
            write_json(output / "geometry-audit.json", geometry_audit(initial.cpu().numpy(), config["physics"]["grid"],
                       config["physics"]["domain_size"], asset_info["occupied_volume"]))
            runtime = build_runtime(config, initial, volumes, beta)
            if case.get("constitutive_adapter") == "scaled-nacc-ray-hardening-v2":
                from wmfracture.experimental_nacc import install_for_probe
                install_for_probe(runtime.sim)
            elif case.get("constitutive_adapter"):
                raise ValueError("Unknown constitutive adapter")
            if case.get("audit_each_step"):
                install_failure_audit(runtime.sim, output)
            appearance = dict(initial=initial, volumes=volumes, covariance=covariance, opacity=opacity, colors=colors,
                              binding=binding, offsets=offsets, source_ids=source_ids, chosen=chosen,
                              beta=torch.as_tensor(beta, device="cuda"), reinforced=torch.as_tensor(reinforced, device="cuda"))
            renderer = SimulationRenderer(config, appearance)
            mass = volumes.cpu().numpy()
            dx = runtime.sim.mpm_model.dx
            rows = {str(factor): [] for factor in (1., 1.5, 2.)}
            trackers = {key: ComponentTracker(mass) for key in rows}
            metrics, image_frames = [], []
            report.update(physics_particles=len(initial), render_gaussians=len(binding), dx=dx)
            with h5py.File(output / "sampled-trajectory.h5", "x") as trajectory:
                trajectory.attrs["resumable"] = False
                trajectory.create_dataset("volumes", data=mass)
                for frame in range(specification["frames"] + 1):
                    if frame:
                        runtime.advance_frame()
                    state = runtime.sim.mpm_state
                    jp = wp.to_torch(state.particle_Jp)
                    x = runtime.sim.export_particle_x_to_torch()
                    v = runtime.sim.export_particle_v_to_torch()
                    trial = wp.to_torch(state.particle_F_trial)
                    if (not all(torch.isfinite(value).all() for value in (x, v, jp, trial, runtime.total_F_tensor))
                        or bool(wp.to_torch(runtime.invalid_F).any())):
                        raise RuntimeError(f"Invalid physical/deformation state at frame {frame}")
                    metrics.append({"frame": frame, "time": float(runtime.sim.time), "Jp_min": float(jp.min()),
                        "Jp_max": float(jp.max()), "Jp_above_render_deletion_threshold_fraction": float((jp > .4).float().mean()),
                        "max_speed": float(v.norm(dim=1).max()), "trial_det_min": float(torch.linalg.det(trial).min())})
                    if frame % specification["analysis_stride"] == 0 or frame == specification["frames"]:
                        positions, velocity = x.cpu().numpy(), v.cpu().numpy()
                        group = trajectory.create_group(f"frames/{frame:05d}")
                        for key, values in (("positions", positions), ("velocities", velocity), ("Jp", jp.cpu().numpy())):
                            group.create_dataset(key, data=values, compression="gzip", compression_opts=1)
                        group.attrs["time"] = runtime.sim.time
                        if case.get("connectivity", True):
                            for key in rows:
                                labels, diagnostic = spatial_components(positions, mass, velocity, dx * float(key))
                                trackers[key].update(labels, diagnostic["major_components"], runtime.sim.time)
                                diagnostic.update(frame=frame, time=float(runtime.sim.time))
                                diagnostic["separation_candidate"] = persistent_motion_candidate(
                                    diagnostic["major_components"], diagnostic["two_largest_relative_speed"])
                                rows[key].append(diagnostic)
                    if frame % specification["render_stride"] == 0 or frame == specification["frames"]:
                        rgb = (renderer.image(runtime).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                        Image.fromarray(rgb).save(output / "frames" / f"{frame:05d}.png")
                        image_frames.append(frame)
                        print(f"{identifier} frame={frame} t={runtime.sim.time:.3f} particles={len(initial)} Jp=[{float(jp.min()):.3g},{float(jp.max()):.3g}]", flush=True)
                    report["last_complete_frame"] = frame
            initially = {key: data[0]["largest_mass_fraction"] >= .98 for key, data in rows.items() if data}
            consensus = simultaneous_consensus(rows, initially) if initially else []
            write_json(output / "separation.json", {"rows": rows, "initially_connected": initially,
                       "consensus_frames": consensus, "assessed": case.get("connectivity", True)})
            write_json(output / "metrics.json", metrics)
            report.update(status="completed", simulation_seconds=float(runtime.sim.time), substeps=runtime.global_step,
                image_frames=image_frames, particle_separation_candidate=bool(consensus) if initially else None,
                max_Jp_over_all_frames=max(row["Jp_max"] for row in metrics),
                opacity_deletion_was_enabled=False)
        except Exception as error:
            report.update(status="failed", error=repr(error))
            if "metrics" in locals():
                write_json(output / "metrics.json", metrics)
            if "runtime" in locals():
                state = runtime.sim.mpm_state
                arrays = {key: wp.to_torch(getattr(state, key)) for key in
                    ("particle_x", "particle_v", "particle_F", "particle_F_trial", "particle_Jp", "particle_stress")}
                arrays.update(total_G=runtime.total_F_tensor, invalid_G=wp.to_torch(runtime.invalid_F))
                diagnostic = {key: int((~torch.isfinite(value)).reshape(len(initial), -1).any(1).sum())
                              for key, value in arrays.items()}
                det = torch.linalg.det(wp.to_torch(state.particle_F))
                diagnostic.update(invalid_G_count=int(arrays["invalid_G"].sum()),
                                  elastic_abs_det_min=float(det.abs().nan_to_num().min()))
                write_json(output / "failure-diagnostic.json", diagnostic)
                bad = arrays["invalid_G"].bool().clone()
                for value in arrays.values():
                    bad |= (~torch.isfinite(value)).reshape(len(initial), -1).any(1)
                ids = bad.nonzero().flatten()[:64]
                if len(ids):
                    np.savez_compressed(output / "failure-state.npz", particle_ids=ids.cpu().numpy(),
                                        **{key: value[ids].cpu().numpy() for key, value in arrays.items()})
            print(f"{identifier} FAILED {error!r}", flush=True)
        report["wall_seconds"] = time.perf_counter() - start
        report["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    report["whole_device_peak_mib"] = max(monitor.samples) if monitor.samples else None
    write_json(output / "summary.json", report)


def summarize(root):
    specification = json.loads((root / "probe-config.json").read_text())
    summaries = {case["id"]: json.loads((root / case["id"] / "summary.json").read_text()) for case in specification["cases"]}
    report = {"status": "audit_completed", "run_directory": str(root), "specification": specification,
              "summaries": summaries, "fracture_accuracy_validated": False,
              "gate_b": "not_validated", "new_algorithm_implemented": False,
              "limitations": ["Source-nearer combines several changes and is not an isolated-factor or exact reproduction.",
                "All-Gaussian case has no connectivity analysis because the dense point graph can exceed host memory.",
                "Sparse temporal and radius-dependent connectivity alone does not validate material cracks.",
                "Whole-device memory includes other applications; wall time includes geometry audit and rendering.",
                "No opacity deletion, particle deletion, crack initialization or state reset was used."]}
    write_json(root / "report.json", report)
    size = specification["render_width"]
    montage = Image.new("RGB", (4 * size, 2 * (size + 54)), "white")
    draw = ImageDraw.Draw(montage)
    for index, (identifier, result) in enumerate(summaries.items()):
        x, y = (index % 4) * size, (index // 4) * (size + 54)
        draw.text((x + 5, y + 4), identifier, fill="black")
        draw.text((x + 5, y + 18), f"{result['status']} | frame {result['last_complete_frame']}", fill="black")
        draw.text((x + 5, y + 32), f"separation: {result.get('particle_separation_candidate', 'unassessed')}", fill="black")
        frames = sorted((root / identifier / "frames").glob("*.png"))
        if frames:
            with Image.open(frames[-1]) as image:
                montage.paste(image, (x, y + 54))
    montage.save(root / "comparison.png")
    artifacts = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts/fracture-probe" / root.name
    artifacts.mkdir(parents=True, exist_ok=True)
    for filename in ("report.json", "comparison.png", "probe-config.json"):
        shutil.copy2(root / filename, artifacts / filename)
    for case in specification["cases"]:
        for filename in ("geometry-audit.json", "first-nonfinite-step.json", "failure-diagnostic.json"):
            source = root / case["id"] / filename
            if source.exists():
                shutil.copy2(source, artifacts / (case["id"] + "-" + filename))
    write_json(artifacts.parent / "latest.json", {"artifact_directory": str(artifacts), "run_directory": str(root)})
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    project = Path(os.environ["WM_PROJECT_ROOT"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=project / "config/fracture-probe.json")
    parser.add_argument("--run", type=Path, help="Analyze a completed audit, or root for internal case execution")
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.case:
        run_case(args.run, args.case)
        return
    if args.run:
        summarize(args.run)
        return
    path = args.config if args.config.is_absolute() else project / args.config
    specification = json.loads(path.read_text())
    config = json.loads((project / specification["base_config"]).read_text())
    config["physics"]["frames"] = specification["frames"]
    config["render"]["width"] = specification["render_width"]
    root = Path(os.environ["WM_RESEARCH_ROOT"]) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-fracture-probe")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "probe-config.json", specification)
    write_json(root / "base-config.json", config)
    write_json(root / "local-code.json", {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), project / "scripts/first_impact.py", *sorted((project / "wmfracture").glob("*.py"))]})
    from wmfracture.checkpoint import physics_fingerprint
    write_json(root / "physics-code.json", physics_fingerprint())
    snapshot = root / "source-snapshot"
    for path in [Path(__file__), project / "scripts/first_impact.py", *sorted((project / "wmfracture").glob("*.py"))]:
        destination = snapshot / path.relative_to(project)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print(f"OUTPUT={root}", flush=True)
    for case in specification["cases"]:
        command = [sys.executable, "-u", str(Path(__file__)), "--run", str(root), "--case", case["id"]]
        with (root / (case["id"] + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            tail = []
            for line in process.stdout:
                log.write(line)
                tail = (tail + [line])[-15:]
                if " frame=" in line or " FAILED " in line:
                    print(line, end="", flush=True)
            if process.wait():
                print("".join(tail), flush=True)
                raise subprocess.CalledProcessError(process.returncode, command)
    summarize(root)


if __name__ == "__main__":
    main()
