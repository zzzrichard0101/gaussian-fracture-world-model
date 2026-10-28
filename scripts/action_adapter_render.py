"""Decode sealed reduced trajectories through the existing Gaussian renderer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from cohesive_probe import BlockObservations


def render(root):
    config = json.loads((root/"config.json").read_text())
    model = json.loads((root/"known-model.json").read_text())
    demo = json.loads((root/"demo.json").read_text())
    with np.load(root/"demo-trajectories.npz", allow_pickle=False) as archive:
        trajectories = {key: archive[key] for key in archive.files}
    observer = BlockObservations(model)
    frames = root/"demo-frames"
    frames.mkdir(exist_ok=True)
    width = config["render_width"]
    mass = model["mass_left"]+model["mass_right"]
    panels = [("truth", "Synthetic reference"), ("baseline", "RGB-D + physics"),
              ("raw", "Learned action head"), ("projected", "Head + physics projection")]
    for index, elapsed in enumerate(config["target_times"]):
        # Known impulse schedule gives exact COM for the two known masses.
        # Only relative separation is decoded from the learned head.
        center = model["mass_left"]/mass * (demo["probe"]*max(0, elapsed-config["probe_time"])
                    + demo["impact"]*max(0, elapsed-config["future_time"]))
        canvas = Image.new("RGB", (4*width, width+64), "white")
        draw = ImageDraw.Draw(canvas)
        for panel_index, (key, label) in enumerate(panels):
            gap = float(trajectories[key][index])
            if np.isfinite(gap):
                positions = {"x_left": center-model["mass_right"]/mass*(model["width"]+gap),
                             "x_right": center+model["mass_left"]/mass*(model["width"]+gap)}
                rgb = observer.capture(positions)["rgb"]
                canvas.paste(Image.fromarray(np.round(np.clip(rgb, 0, 1)*255).astype(np.uint8)), (panel_index*width, 0))
            else:
                label += " (no match)"
            draw.text((panel_index*width+7, width+4), label, fill="black")
            draw.text((panel_index*width+7, width+21), f"gap={gap*1000:.2f} mm", fill="black")
        draw.text((7, width+43), f"t={elapsed:.3f}s; known new velocity increment={demo['impact']:.2f}m/s; 1D fixture, slow playback", fill="black")
        canvas.save(frames/f"{index:05d}.png")
    canvas.save(root/"prediction-end.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i",
        str(frames/"%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(root/"prediction-comparison.mp4")], check=True)
    (root/"render-audit.json").write_text(json.dumps({"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "demo_rule": "First probed test history, action 0.24 m/s, V-JEPA head seed 17; not selected by accuracy.",
        "geometry": "Fixed Gaussian colors/covariances, analytic known COM, predicted relative gap.",
        "raw_head_physical_feasibility_guaranteed": False,
        "min_gap_m": {key: float(np.min(value)) for key, value in trajectories.items()}}, indent=2))
    target = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/action-adapter"
    for filename in ("prediction-end.png", "prediction-comparison.mp4", "render-audit.json"):
        shutil.copy2(root/filename, target/filename)
    print(f"GAUSSIAN_VIDEO={root/'prediction-comparison.mp4'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    render(parser.parse_args().run)
