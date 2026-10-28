"""Synthetic Gaussian RGB/alpha/expected-depth observations.

Depth is alpha-normalized camera-z of Gaussian centers, not a calibrated
surface-depth sensor. Floor occlusion participates in the same compositing.
"""
import copy

import torch
import warp as wp

from wmfracture.geometry import advect_gaussians
from wmfracture.rendering import build_renderer, floor_gaussians


class GaussianRGBD:
    def __init__(self, display):
        self.rgb_rasterizer, self.camera = build_renderer(display)
        black = copy.deepcopy(display)
        black["background"] = [0., 0., 0.]
        self.field_rasterizer, _ = build_renderer(black)
        self.world_to_camera = torch.tensor(self.camera.world_to_camera, device="cuda")

    @torch.no_grad()
    def points(self, positions, covariance, colors, opacity, object_count):
        kwargs = dict(means3D=positions, means2D=torch.zeros_like(positions),
                      opacities=opacity, cov3D_precomp=covariance)
        rgb, _, _ = self.rgb_rasterizer(colors_precomp=colors, **kwargs)
        z = positions @ self.world_to_camera[2, :3] + self.world_to_camera[2, 3]
        fields = torch.stack([z, torch.ones_like(z),
                              (torch.arange(len(z), device=z.device) < object_count).float()], dim=1)
        rendered, _, _ = self.field_rasterizer(colors_precomp=fields, **kwargs)
        alpha = rendered[1].clamp(0, 1)
        depth = torch.where(alpha > 1e-6, rendered[0] / alpha.clamp_min(1e-6), torch.zeros_like(alpha))
        object_alpha = rendered[2].clamp(0, 1)
        if not all(torch.isfinite(tensor).all() for tensor in (rgb, depth, alpha, object_alpha)):
            raise RuntimeError("Non-finite synthetic observation")
        return {"rgb": rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
                "depth": depth.cpu().numpy(), "alpha": alpha.cpu().numpy(),
                "object_alpha": object_alpha.cpu().numpy()}

    @torch.no_grad()
    def capture(self, runtime, appearance, floor):
        affine = (runtime.total_F_tensor if runtime.config["render"].get("deformation", "trial") == "total"
                  else wp.to_torch(runtime.sim.mpm_state.particle_F_trial))
        points, cov = advect_gaussians(runtime.sim.export_particle_x_to_torch(), affine,
            appearance["binding"], appearance["offsets"], appearance["covariance"])
        return self.points(torch.cat([points, floor[0]]), torch.cat([cov, floor[1]]),
                           torch.cat([appearance["colors"], floor[2]]),
                           torch.cat([appearance["opacity"], floor[3]]), len(points))
