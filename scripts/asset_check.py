"""Render a released Gaussian asset from its calibrated camera, without simulation."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import focal2fov


def main():
    root = Path(os.environ["WM_RESEARCH_ROOT"]) / "data/gaussianfluent/model/watermelon"
    output = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    ply = root / "point_cloud/iteration_30000/point_cloud.ply"
    if not ply.is_file():
        raise FileNotFoundError("Run fetch-asset first")
    camera_data = json.loads((root / "cameras.json").read_text())[0]
    rotation = np.array(camera_data["rotation"], dtype=np.float32)
    position = np.array(camera_data["position"], dtype=np.float32)
    width = 512
    height = round(width * camera_data["height"] / camera_data["width"])
    camera = Camera(
        colmap_id=camera_data["id"], uid=0, R=rotation, T=-rotation.T @ position,
        FoVx=focal2fov(camera_data["fx"], camera_data["width"]),
        FoVy=focal2fov(camera_data["fy"], camera_data["height"]),
        image=torch.zeros((3, height, width)), gt_alpha_mask=None,
        image_name=camera_data["img_name"],
    )
    model = GaussianModel(3)
    model.load_ply(str(ply))
    settings = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        result = render(camera, model, settings, torch.ones(3, device="cuda"))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    image = result["render"].clamp(0, 1)
    assert torch.isfinite(image).all() and image.std() > 0.05
    assert result["visibility_filter"].any()
    pixels = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(pixels).save(output / "watermelon-preview.png")
    report = {
        "passed": True, "asset": "watermelon", "gaussians": len(model.get_xyz),
        "visible_gaussians": int(result["visibility_filter"].sum()),
        "resolution": [width, height], "render_wall_seconds_first_call": elapsed,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "scope": "Static released asset only; no fracture or physics prediction is validated.",
    }
    (output / "asset-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
