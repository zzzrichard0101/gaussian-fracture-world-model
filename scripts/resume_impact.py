"""Resume physics in a fresh process using only the checkpoint and installed code."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import h5py
import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw

from wmfracture.checkpoint import load_checkpoint, save_checkpoint
from wmfracture.rendering import SimulationRenderer


def reset_hidden_state(runtime):
    """Ablation only: retain x/v and physical parameters, erase latent history."""
    state = runtime.sim.mpm_state
    identity = torch.eye(3, device="cuda").repeat(runtime.sim.n_particles, 1, 1)
    wp.to_torch(state.particle_F).copy_(identity)
    wp.to_torch(state.particle_F_trial).copy_(identity)
    for name in ("particle_C", "particle_stress", "particle_R"):
        wp.to_torch(getattr(state, name)).zero_()
    wp.to_torch(state.particle_Jp).fill_(runtime.config["physics"]["material"].get("alpha_0", -.04))
    wp.to_torch(state.particle_cov).copy_(wp.to_torch(state.particle_init_cov))
    runtime.total_F_tensor.copy_(identity)
    wp.to_torch(runtime.invalid_F).zero_()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--until-frame", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reset-hidden-state", action="store_true", help="Deliberately incomplete-state ablation")
    args = parser.parse_args()
    loaded = load_checkpoint(args.checkpoint)
    runtime, appearance = loaded.runtime, loaded.appearance
    config = runtime.config
    until = config["physics"]["frames"] if args.until_frame is None else args.until_frame
    start_frame = runtime.frame_index
    if until <= start_frame:
        raise ValueError("until-frame must be later than the checkpoint")
    output = args.output or (Path(os.environ["WM_RESEARCH_ROOT"]) / "runs" /
                            (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-resumed"))
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT={output}", flush=True)
    audit = {"status": "restored-exact", "checkpoint": str(args.checkpoint), "verified_arrays": loaded.verified_arrays,
             "frame_index": start_frame, "time": runtime.sim.time, "global_step": runtime.global_step,
             "scalars_and_boundaries_verified": True, "external_asset_read": False,
             "reset_hidden_state_ablation": args.reset_hidden_state}
    (output / "restore-audit.json").write_text(json.dumps(audit, indent=2))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    if args.reset_hidden_state:
        reset_hidden_state(runtime)
    renderer = SimulationRenderer(config, appearance)
    frames = output / "frames"
    frames.mkdir()
    with h5py.File(output / "trajectory.h5", "w") as trajectory:
        trajectory.attrs["resumable_checkpoint"] = False
        trajectory.attrs["start_frame"] = start_frame
        trajectory.create_dataset("initial_positions", data=appearance["initial"].cpu().numpy())
        trajectory.create_dataset("volumes", data=appearance["volumes"].cpu().numpy())
        for frame in range(start_frame, until + 1):
            if frame > start_frame:
                runtime.advance_frame()
            state = runtime.sim.mpm_state
            fields = {"positions": wp.to_torch(state.particle_x), "velocities": wp.to_torch(state.particle_v),
                      "Jp": wp.to_torch(state.particle_Jp), "trial_F": wp.to_torch(state.particle_F_trial),
                      "total_F": runtime.total_F_tensor}
            if not all(torch.isfinite(value).all() for value in fields.values()) or wp.to_torch(runtime.invalid_F).any():
                raise RuntimeError(f"Invalid resumed state at frame {frame}")
            group = trajectory.create_group(f"frames/{frame:05d}")
            group.attrs["time"] = runtime.sim.time
            group.attrs["global_step"] = runtime.global_step
            for name, value in fields.items():
                group.create_dataset(name, data=value.cpu().numpy(), compression="gzip", compression_opts=1)
            pixels = (renderer.image(runtime).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(pixels)
            ImageDraw.Draw(image).text((12, 12), f"Resumed | frame={frame} | t={runtime.sim.time:.3f}s", fill=(20, 20, 20))
            image.save(frames / f"{frame:05d}.png")
            print(f"resumed frame={frame} time={runtime.sim.time:.3f} steps={runtime.global_step}", flush=True)
    final_path = output / "checkpoint-final.h5"
    save_checkpoint(final_path, runtime, appearance, provenance={"parent_checkpoint": str(args.checkpoint),
                    "parent_provenance": loaded.metadata["provenance"], "reset_hidden_state_ablation": args.reset_hidden_state})
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", str(config["render"]["fps"]),
                    "-start_number", str(start_frame), "-i", str(frames / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(output / "resumed.mp4")], check=True)
    result = {"status": "completed", "start_frame": start_frame, "end_frame": until,
              "final_time": runtime.sim.time, "final_global_step": runtime.global_step,
              "checkpoint_final": str(final_path), "restore_audit": audit,
              "note": "Physics continuation, not fracture-accuracy validation."}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
