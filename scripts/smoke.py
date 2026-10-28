"""Small GPU integration checks, not a fracture-accuracy benchmark."""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def rasterizer_check(output):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from simple_knn._C import distCUDA2
    from utils.graphics_utils import getProjectionMatrix

    points = torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [0.0, 0.2, 2.0], [-0.2, -0.1, 2.0]], device="cuda")
    distances = distCUDA2(points)
    assert torch.isfinite(distances).all() and (distances > 0).all(), "KNN returned invalid distances"
    colors = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.8, 0.2], [0.1, 0.3, 0.9], [0.9, 0.7, 0.1]], device="cuda", requires_grad=True)
    projection = getProjectionMatrix(0.01, 10.0, math.pi / 3, math.pi / 3).transpose(0, 1).cuda()
    settings = GaussianRasterizationSettings(
        image_height=128, image_width=128, tanfovx=math.tan(math.pi / 6), tanfovy=math.tan(math.pi / 6),
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"), projmatrix=projection,
        sh_degree=0, campos=torch.zeros(3, device="cuda"), prefiltered=False, debug=False,
    )
    rendered, radii, point_xy = GaussianRasterizer(settings)(
        means3D=points, means2D=torch.zeros_like(points, requires_grad=True),
        opacities=torch.full((4, 1), 0.9, device="cuda"), colors_precomp=colors,
        scales=torch.full((4, 3), 0.08, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, device="cuda"),
    )
    assert rendered.shape == (3, 128, 128) and torch.isfinite(rendered).all()
    assert (radii > 0).any() and rendered.sum().item() > 1.0, "Rasterizer produced an empty image"
    rendered.sum().backward()
    assert colors.grad is not None and torch.isfinite(colors.grad).all() and colors.grad.abs().sum() > 0
    pixels = (rendered.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(pixels).save(output / "rasterizer-smoke.png")
    return {"passed": True, "visible_gaussians": int((radii > 0).sum()), "point_xy_shape": list(point_xy.shape), "backward_passed": True}


def mpm_check(output):
    import warp as wp
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP

    wp.init()
    axis = torch.linspace(0.42, 0.52, 6, device="cuda")
    points = torch.cartesian_prod(axis, axis + 0.15, axis).contiguous()
    volumes = torch.full((len(points),), 0.02 ** 3, device="cuda")
    sim = MPM_Simulator_WARP(len(points), n_grid=32, grid_lim=1.0)
    sim.load_initial_data_from_torch(points, volumes, n_grid=32, grid_lim=1.0)
    # Call the dictionary method directly: upstream's convenience wrapper swaps its arguments.
    sim.set_parameters_dict({"material": "jelly", "E": 1000.0, "nu": 0.2, "density": 1000.0, "g": [0.0, -9.81, 0.0], "beta": 1.0})
    sim.finalize_mu_lam()
    initial = sim.export_particle_x_to_torch().clone()
    steps, dt = 100, 0.0002
    started = time.perf_counter()
    for step in range(steps):
        sim.p2g2p(step, dt, flip_pic=False)
    wp.synchronize()
    positions = sim.export_particle_x_to_torch().clone()
    velocities = sim.export_particle_v_to_torch().clone()
    displacement = (positions - initial).mean(dim=0)
    assert torch.isfinite(positions).all() and torch.isfinite(velocities).all(), "MPM produced NaN/Inf"
    assert -0.01 < displacement[1].item() < -0.0001, "MPM gravity did not move the block as expected"
    assert velocities[:, 1].mean().item() < -0.01, "MPM gravity velocity has the wrong sign"
    np.savez_compressed(output / "mpm-smoke.npz", initial=initial.cpu().numpy(), positions=positions.cpu().numpy(), velocities=velocities.cpu().numpy())
    return {"passed": True, "material": "jelly", "particles": len(points), "grid": 32, "steps": steps, "simulated_seconds": steps * dt,
            "mean_displacement": displacement.cpu().tolist(), "integration_wall_seconds": time.perf_counter() - started,
            "scope": "Free-fall integration check; fracture, checkpoint recovery, and inverse fitting are not tested."}


def taichi_check():
    import taichi as ti

    ti.init(arch=ti.cuda, enable_fallback=False, device_memory_GB=0.25, random_seed=42)
    values = ti.field(dtype=ti.f32, shape=32)

    @ti.kernel
    def fill():
        for i in values:
            values[i] = i * 2.0

    fill()
    np.testing.assert_array_equal(values.to_numpy(), np.arange(32, dtype=np.float32) * 2)
    assert ti.lang.impl.current_cfg().arch == ti.cuda
    import particle_filling.filling
    import open3d
    import pymeshlab

    ti.reset()
    return {"passed": True, "backend": "cuda", "filling_module_import": True,
            "open3d_version": open3d.__version__, "pymeshlab_import": True}


def main():
    output = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.reset_peak_memory_stats()
    report = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda}
    report["rasterizer"] = rasterizer_check(output)
    report["mpm"] = mpm_check(output)
    report["taichi"] = taichi_check()
    report["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    report["memory_note"] = "PyTorch allocation only; Warp allocations and other GPU processes are excluded."
    (output / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
