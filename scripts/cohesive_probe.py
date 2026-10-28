"""Small cohesive/contact verification and Gaussian observation pipeline.

Predetermined single interface, one translation per block. This fixture cannot
validate arbitrary 3D fracture paths, MPM grid contact, or novel inverse models.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from wmfracture.cohesive import InterfaceLaw, InterfaceState, TwoBlockSystem, prescribed_cycle, save_state


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def tick(value, dt):
    result = round(value / dt)
    if abs(result * dt - value) > 1e-12:
        raise ValueError("All scheduled times and output intervals must align with dt")
    return result


def rollout(config, case, dt, output):
    output.mkdir(exist_ok=False)
    law = InterfaceLaw(**config["law"])
    system = TwoBlockSystem(law, InterfaceState(-config["width"] / 2, config["width"] / 2),
                            config["mass_left"], config["mass_right"], config["width"])
    history = prescribed_cycle(system, case["prehistory_peak"], config["prehistory_samples"], config["prehistory_duration"])
    save_state(output / "release-state.json", system)
    write_json(output / "prehistory.json", {"prescribed_motion": "delta=peak*sin(pi*t/duration)^2",
        "duration": config["prehistory_duration"], "records": history,
        "actuator_work": system.state.external_work, "release_state": asdict(system.state)})
    initial_momentum = system.momentum()
    initial_residual = system.energies()["accounted_total"] - system.state.external_work
    expected_momentum = initial_momentum
    events = {}
    for event in case["impulses"]:
        events.setdefault(tick(event["time"], dt), []).append(event)
    count, stride = tick(config["duration"], dt), tick(config["sample_interval"], dt)
    rows, event_log = [], []
    first_failure = None
    post_failure_contacts = 0
    max_energy_residual = 0.
    max_momentum_residual = 0.
    previous_damage = law.damage(system.state.maximum_opening)
    damage_monotone = True
    for index in range(count + 1):
        # State at a tick is observed before impulses at that tick, matching
        # the frame-boundary convention used by the existing MPM pipeline.
        energies = system.energies()
        damage = law.damage(system.state.maximum_opening)
        if damage == 1. and first_failure is None:
            first_failure = index * dt
        if damage == 1. and system.opening < 0:
            post_failure_contacts += 1
        damage_monotone &= damage >= previous_damage
        previous_damage = damage
        max_energy_residual = max(max_energy_residual, abs(energies["accounted_total"] - system.state.external_work))
        max_momentum_residual = max(max_momentum_residual, abs(system.momentum() - expected_momentum))
        if index % stride == 0 or index == count:
            rows.append({"elapsed": index * dt, **asdict(system.state), "opening": system.opening,
                "damage": damage, "traction": law.traction(system.opening, system.state.maximum_opening),
                **energies, "momentum": system.momentum()})
        for event in events.get(index, []):
            audit = system.impulse(event["left"], event["right"])
            expected_momentum += audit["momentum_increment"]
            event_log.append({"id": event["id"], **audit})
        if index < count:
            system.advance(dt)
    energy_scale = max(law.area * law.fracture_energy,
                       max(abs(row["external_work"]) for row in rows))
    summary = {"status": "completed", "case": case, "dt": dt, "steps": count,
        "first_failure_elapsed": first_failure, "post_failure_contact_steps": post_failure_contacts,
        "damage_monotone": damage_monotone, "final_damage": rows[-1]["damage"], "final_gap_m": rows[-1]["opening"],
        "max_energy_residual_j": max_energy_residual, "energy_scale_j": energy_scale,
        "prehistory_energy_residual_j": initial_residual,
        "max_relative_energy_residual": max_energy_residual / energy_scale,
        "max_momentum_residual": max_momentum_residual, "minimum_sampled_gap_m": min(row["opening"] for row in rows),
        "event_log": event_log, "release_state": json.loads((output / "release-state.json").read_text())["payload"]["state"]}
    write_json(output / "summary.json", summary)
    with h5py.File(output / "evaluation.h5", "x") as stream:
        stream.attrs["scope"] = "Privileged physical state, future response and prehistory; not inverse-model input."
        for key in rows[0]:
            stream.create_dataset(key, data=np.array([row[key] for row in rows]))
    print(f"{case['id']} dt={dt:g} broken={summary['final_damage']==1} energy_residual={summary['max_relative_energy_residual']:.3g}", flush=True)
    return rows, summary


class BlockObservations:
    def __init__(self, config):
        import torch
        from wmfracture.observations import GaussianRGBD
        from wmfracture.rendering import floor_gaussians
        self.torch, self.config = torch, config
        self.renderer = GaussianRGBD(config["render"])
        n = config["render"]["gaussians_per_axis"]
        width, side = config["width"], math.sqrt(config["law"]["area"])
        spacing = np.array([width, side, side]) / n
        axes = [torch.linspace(-extent/2 + step/2, extent/2 - step/2, n, device="cuda")
                for extent, step in zip([width, side, side], spacing)]
        points = torch.cartesian_prod(*axes)
        self.local = points.repeat(2, 1)
        self.count = len(points)
        checker = ((torch.arange(self.count, device="cuda") // n) % 2).float()
        palettes = [[.16, .42, .72], [.85, .42, .15]]
        self.colors = torch.cat([(torch.tensor(color, device="cuda")[None] * (.82 + .18 * checker[:, None])) for color in palettes])
        variance = (spacing * .60) ** 2
        self.cov = torch.tensor([variance[0], 0., 0., variance[1], 0., variance[2]], dtype=torch.float32, device="cuda").repeat(len(self.local), 1)
        self.opacity = torch.full((len(self.local), 1), .98, device="cuda")
        self.floor = floor_gaussians(.75 - side / 2, 2.)

    def capture(self, row):
        torch = self.torch
        with torch.no_grad():
            positions = self.local.clone()
            positions[:self.count, 0] += row["x_left"] + 1.
            positions[self.count:, 0] += row["x_right"] + 1.
            positions[:, 1] += 1.
            positions[:, 2] += .75
            return self.renderer.points(torch.cat([positions, self.floor[0]]), torch.cat([self.cov, self.floor[1]]),
                torch.cat([self.colors, self.floor[2]]), torch.cat([self.opacity, self.floor[3]]), len(positions))


def render(config, root, results, level):
    observer = BlockObservations(config)
    sample_dt = config["sample_interval"]
    stride = tick(config["render_interval"], sample_dt)
    indices = list(range(0, len(results[0]["rows"][level]), stride))
    observation_indices = {tick(value, sample_dt) for value in config["observation_times"]}
    observation_arrays = {}
    width = config["render"]["width"]
    output_frames = root / "video-frames"
    output_frames.mkdir()
    handles = [h5py.File(root / case["id"] / "observations.h5", "x") for case in config["cases"]]
    try:
        for handle in handles:
            camera = observer.renderer.camera
            handle.attrs["camera"] = json.dumps({"world_to_camera": camera.world_to_camera.tolist(),
                "width": camera.width, "height": camera.height, "fovx": camera.fovx, "fovy": camera.fovy})
            handle.attrs["scope"] = "Raw RGB, alpha-normalized Gaussian-center depth and alpha. No damage display or future response."
        selected = sorted(set(indices) | observation_indices)
        sequence = 0
        for index in selected:
            combined = Image.new("RGB", (width * len(results), width + 62), "white")
            draw = ImageDraw.Draw(combined)
            for column, (result, handle) in enumerate(zip(results, handles)):
                row = result["rows"][level][index]
                fields = observer.capture(row)
                raw = Image.fromarray((fields["rgb"] * 255).round().astype(np.uint8))
                if index in observation_indices:
                    group = handle.create_group(f"frames/{index:05d}")
                    group.attrs["elapsed"] = row["elapsed"]
                    for key, array in fields.items():
                        group.create_dataset(key, data=array, compression="gzip", compression_opts=1)
                    if column < 2:
                        observation_arrays[(column, index)] = fields
                if index in indices:
                    combined.paste(raw, (column * width, 62))
                    draw.text((column * width + 8, 6), f"{result['id']} | 1D cohesive interface", fill="black")
                    draw.text((column * width + 8, 22), f"t={row['elapsed']:.3f}s | D={row['damage']:.4f} | gap={row['opening']*1000:.2f}mm", fill="black")
                    draw.text((column * width + 8, 38), "Overlay uses privileged state; raw observations do not.", fill="black")
            if index in indices:
                combined.save(output_frames / f"{sequence:05d}.png")
                sequence += 1
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(config["render"]["fps"]),
            "-i", str(output_frames / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(root / "cohesive-contact.mp4")], check=True)
        raw_difference = {key: max(float(np.max(np.abs(observation_arrays[(0, index)][key] - observation_arrays[(1, index)][key])))
                                   for index in observation_indices) for key in ("rgb", "depth", "object_alpha")}
        with Image.open(output_frames / "00000.png") as before, Image.open(output_frames / f"{sequence-1:05d}.png") as after:
            montage = Image.new("RGB", (before.width, before.height * 2), "white")
            montage.paste(before, (0, 0)); montage.paste(after, (0, before.height))
            montage.save(root / "before-after.png")
        return {"raw_observation_max_absolute_difference": raw_difference,
                "render_level": level, "render_dt": config["dt_levels"][level],
                "gaussians_per_block": observer.count, "opacity_deletion_enabled": False}
    finally:
        for handle in handles:
            handle.close()


def main():
    project = Path(os.environ["WM_PROJECT_ROOT"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=project / "config/cohesive-probe.json")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else project / args.config
    config = json.loads(path.read_text())
    for dt in config["dt_levels"]:
        for value in [config["duration"], config["sample_interval"], *config["observation_times"]]:
            tick(value, dt)
    root = Path(os.environ["WM_RESEARCH_ROOT"]) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-cohesive-probe")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    write_json(root / "code-sha256.json", {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), project / "wmfracture/cohesive.py", project / "wmfracture/observations.py", project / "wmfracture/rendering.py"]})
    print(f"OUTPUT={root}", flush=True)
    started = time.perf_counter()
    results = []
    for case in config["cases"]:
        folder = root / case["id"]
        folder.mkdir()
        result = {"id": case["id"], "rows": [], "summaries": []}
        for level, dt in enumerate(config["dt_levels"]):
            rows, summary = rollout(config, case, dt, folder / f"level-{level}")
            result["rows"].append(rows)
            result["summaries"].append(summary)
        results.append(result)
    comparisons = {}
    for result in results:
        pairs = []
        for level, (first, second) in enumerate(zip(result["rows"][:-1], result["rows"][1:])):
            position = max(abs(a[key] - b[key]) for a, b in zip(first, second) for key in ("x_left", "x_right"))
            times = [result["summaries"][i]["first_failure_elapsed"] for i in (level, level+1)]
            pairs.append({"coarse_dt": config["dt_levels"][level], "fine_dt": config["dt_levels"][level+1],
                          "max_position_difference_m": position,
                          "failure_time_difference_s": abs(times[0] - times[1]) if None not in times else None})
        comparisons[result["id"]] = pairs
    gaussian_report = render(config, root, results, len(config["dt_levels"]) - 1)
    finest = {result["id"]: result["summaries"][-1] for result in results}
    tolerance = config["thresholds"]
    checks = {
        "energy_balance": all(summary["max_relative_energy_residual"] <= tolerance["maximum_relative_energy_residual"]
                              for result in results for summary in result["summaries"]),
        "momentum": all(summary["max_momentum_residual"] <= tolerance["maximum_momentum_residual"]
                        for result in results for summary in result["summaries"]),
        "irreversible_damage": all(summary["damage_monotone"] for result in results for summary in result["summaries"]),
        "finest_pair_position_tolerance": all(pairs[-1]["max_position_difference_m"] <= tolerance["finest_pair_max_position_difference_m"] for pairs in comparisons.values()),
        "same_failure_outcome_across_dt": all(len({summary["final_damage"] == 1. for summary in result["summaries"]}) == 1 for result in results),
        "finest_pair_failure_timing": all(
            ((result["summaries"][-2]["first_failure_elapsed"] is None) == (result["summaries"][-1]["first_failure_elapsed"] is None))
            and (comparisons[result["id"]][-1]["failure_time_difference_s"] is None
                 or comparisons[result["id"]][-1]["failure_time_difference_s"] <= tolerance["finest_pair_failure_time_difference_s"])
            for result in results),
        "same_raw_observations": max(gaussian_report["raw_observation_max_absolute_difference"].values()) <= tolerance["raw_observation_max_difference"],
        "different_future_outcomes": finest["pristine"]["final_damage"] < 1. and finest["preloaded"]["final_damage"] == 1.
            and abs(finest["preloaded"]["final_gap_m"] - finest["pristine"]["final_gap_m"]) >= tolerance["future_pair_final_gap_difference_min_m"],
        "contact_after_failure": finest["break-recontact"]["post_failure_contact_steps"] > 0 and finest["break-recontact"]["final_damage"] == 1.
    }
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for result in results:
        rows = result["rows"][-1]
        elapsed = [row["elapsed"] for row in rows]
        axes[0].plot(elapsed, [row["opening"] * 1000 for row in rows], label=result["id"])
        axes[1].plot(elapsed, [row["damage"] for row in rows], label=result["id"])
        pairs = comparisons[result["id"]]
        axes[2].loglog([pair["fine_dt"] for pair in pairs], [pair["max_position_difference_m"] for pair in pairs], "o-", label=result["id"])
    axes[0].set_ylabel("Interface gap (mm)")
    axes[0].set_title("Negative gap = penalty overlap", fontsize=10)
    axes[1].set_ylabel("Irreversible damage D")
    axes[2].set_ylabel("Max. position difference (m)")
    axes[2].set_title("Adjacent time refinements", fontsize=10)
    axes[0].set_xlabel("Time after release (s)"); axes[1].set_xlabel("Time after release (s)")
    axes[2].set_xlabel("Fine time step (s)")
    for axis in axes:
        axis.grid(alpha=.25); axis.legend(fontsize=7)
    figure.savefig(root / "mechanics.png", dpi=150)
    plt.close(figure)
    report = {"status": "fixture_checks_passed" if all(checks.values()) else "fixture_checks_failed",
        "run_directory": str(root), "config": config, "checks": checks,
        "finest_results": finest, "time_refinement": comparisons, "gaussian_observations": gaussian_report,
        "wall_seconds": time.perf_counter() - started,
        "mpm_fracture_gate_b_passed": False, "inverse_algorithm_implemented": False,
        "limitations": ["One prescribed bonded interface and one translational DOF per rigid block; no arbitrary 3D crack path.",
            "Local dynamics are evaluated in CPU float64; GPU is used for Gaussian observation generation.",
            "Positive fixture result validates this reduced model only; it does not validate the earlier MPM backend.",
            "Existing bilinear cohesive law, ideal guides, finite penalty contact, no friction/rotation/bulk elasticity.",
            "Preset illustrative SI coefficients; no experimental calibration or claim of novel fracture physics.",
            "Same-image hidden-state ambiguity is a known mechanics effect and a positive control, not research novelty.",
            "Dedicated reduced-state JSON supports free-dynamics restart; full scheduled checkpoint integration is not implemented."]}
    write_json(root / "report.json", report)
    artifacts = project / "artifacts/cohesive-probe"
    artifacts.mkdir(parents=True, exist_ok=True)
    for filename in ("report.json", "config.json", "mechanics.png", "cohesive-contact.mp4", "before-after.png"):
        shutil.copy2(root / filename, artifacts / filename)
    print(json.dumps({"status": report["status"], "checks": checks,
                      "final_gaps_m": {key: row["final_gap_m"] for key, row in finest.items()},
                      "observation_difference": gaussian_report["raw_observation_max_absolute_difference"]}, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
