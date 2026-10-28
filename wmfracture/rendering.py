"""Fixed camera and a Gaussian renderer using explicit simulation coordinates."""
import math
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getProjectionMatrix
from wmfracture.geometry import advect_gaussians


def build_renderer(config):
    eye = np.array(config["camera_eye"], dtype=np.float32)
    forward = np.array(config["camera_target"], dtype=np.float32) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 0, 1], dtype=np.float32))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = np.stack([right, down, forward])
    world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ eye
    view = torch.tensor(world_to_camera, device="cuda").T.contiguous()
    width = config["width"]
    height = width
    fovy = math.radians(config["vertical_fov_degrees"])
    fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
    projection = getProjectionMatrix(0.01, 20.0, fovx, fovy).T.cuda()
    settings = GaussianRasterizationSettings(
        image_height=height, image_width=width, tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2),
        bg=torch.tensor(config["background"], device="cuda"), scale_modifier=1.0,
        viewmatrix=view, projmatrix=view @ projection, sh_degree=0,
        campos=torch.tensor(eye, device="cuda"), prefiltered=False, debug=False,
    )
    return GaussianRasterizer(settings), SimpleNamespace(world_to_camera=world_to_camera, fovx=fovx, fovy=fovy,
                                                        width=width, height=height, eye=eye)


def floor_gaussians(height, domain_size):
    """A static checker plane for visual reference; never added to the MPM state."""
    axis = torch.linspace(0, domain_size, 40, device="cuda")
    xy = torch.cartesian_prod(axis, axis)
    positions = torch.cat([xy, torch.full((len(xy), 1), height, device="cuda")], dim=1)
    checker = (torch.floor(xy[:, 0] * 5) + torch.floor(xy[:, 1] * 5)).remainder(2)
    colors = (0.82 + checker[:, None] * 0.10).expand(-1, 3).contiguous()
    covariance = torch.tensor([0.0015, 0, 0, 0.0015, 0, 0.000001], device="cuda").repeat(len(xy), 1)
    opacity = torch.full((len(xy), 1), 0.98, device="cuda")
    return positions, covariance, colors, opacity


class SimulationRenderer:
    def __init__(self, config, appearance):
        self.config, self.appearance = config, appearance
        self.rasterizer, self.camera = build_renderer(config["render"])
        floor = floor_gaussians(config["physics"]["floor"]["height"], config["physics"]["domain_size"])
        self.floor = floor
        self.colors = torch.cat([appearance["colors"], floor[2]])
        self.opacity = torch.cat([appearance["opacity"], floor[3]])

    def image(self, runtime):
        with torch.no_grad():
            affine = (runtime.total_F_tensor if self.config["render"].get("deformation", "trial") == "total"
                      else wp.to_torch(runtime.sim.mpm_state.particle_F_trial))
            appearance = self.appearance
            points, covariance = advect_gaussians(runtime.sim.export_particle_x_to_torch(), affine,
                appearance["binding"], appearance["offsets"], appearance["covariance"])
            points = torch.cat([points, self.floor[0]])
            image, _, _ = self.rasterizer(means3D=points, means2D=torch.zeros_like(points),
                colors_precomp=self.colors, opacities=self.opacity, cov3D_precomp=torch.cat([covariance, self.floor[1]]))
            if not torch.isfinite(image).all():
                raise RuntimeError("Non-finite checkpoint rendering")
            return image.clamp(0, 1)
