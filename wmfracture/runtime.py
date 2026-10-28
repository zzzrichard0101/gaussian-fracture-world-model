"""Shared frame stepping and supported boundary construction."""
from dataclasses import dataclass, field
import copy
import math

import torch
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from wmfracture.solver_adapter import enable_guarded_flip, accumulate_total_deformation
from wmfracture.loading import validate_schedule, apply_impulse


@dataclass
class SimulationRuntime:
    config: dict
    sim: object
    total_F_tensor: torch.Tensor
    total_F: object
    invalid_F: object
    frame_index: int = 0
    global_step: int = 0
    applied_loads: list = field(default_factory=list)

    def advance_frame(self):
        physics = self.config["physics"]
        for event in physics.get("loading_schedule", []):
            if event["frame"] == self.frame_index:
                if event["id"] in {row["id"] for row in self.applied_loads}:
                    raise RuntimeError("Refusing to apply an impulse twice; restart from a frame boundary")
                self.applied_loads.append(apply_impulse(self.sim, event))
        model = self.sim.mpm_model
        E, nu = wp.to_torch(model.E), wp.to_torch(model.nu)
        rho = wp.to_torch(self.sim.mpm_state.particle_density)
        sound_speed = ((E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho)).sqrt()).max().item()
        max_speed = self.sim.export_particle_v_to_torch().norm(dim=1).max().item()
        if not math.isfinite(sound_speed + max_speed):
            raise RuntimeError("Non-finite wave/particle speed")
        safe_dt = min(physics["max_substep_dt"], physics["cfl"] * model.dx / (sound_speed + max_speed))
        steps = math.ceil(physics["frame_dt"] / safe_dt)
        if steps < 1 or steps > 2000:
            raise RuntimeError("Invalid/excessive CFL substep count")
        dt = physics["frame_dt"] / steps
        for _ in range(steps):
            self.sim.p2g2p(self.global_step, dt, flip_pic=physics["transfer"] == "guarded-flip",
                           flip_pic_ratio=physics.get("flip_pic_ratio", .7))
            if self.config["render"].get("deformation", "trial") == "total":
                wp.launch(accumulate_total_deformation, dim=self.sim.n_particles,
                          inputs=[self.sim.mpm_state, self.total_F, self.invalid_F], device="cuda:0")
            self.global_step += 1
            if self.global_step % 25 == 0:
                x, v = self.sim.export_particle_x_to_torch(), self.sim.export_particle_v_to_torch()
                if (not torch.isfinite(x).all() or not torch.isfinite(v).all()
                    or x.min() < .5 * model.dx or x.max() >= model.grid_lim - model.dx):
                    raise RuntimeError(f"Unsafe particle state at step {self.global_step}")
        self.frame_index += 1
        return steps, dt


def build_runtime(config, initial, volumes, beta):
    physics = config["physics"]
    validate_schedule(physics.get("loading_schedule", []))
    if physics["transfer"] not in ("apic", "guarded-flip"):
        raise ValueError("Unsupported transfer")
    if any(physics.get(key) for key in ("impulses", "moving_colliders", "velocity_modifiers")):
        raise ValueError("This runtime supports a static floor, box, and empty-mass guard only")
    sim = MPM_Simulator_WARP(len(initial), n_grid=physics["grid"], grid_lim=physics["domain_size"])
    sim.load_initial_data_from_torch(initial, volumes, n_grid=physics["grid"], grid_lim=physics["domain_size"])
    sim.set_parameters_dict(physics["material"])
    sim.mpm_model.beta.assign(beta.detach().cpu().numpy() if isinstance(beta, torch.Tensor) else beta)
    sim.finalize_mu_lam()
    sim.import_particle_v_from_torch(torch.tensor(physics["initial_velocity"], device="cuda").expand(len(initial), 3).contiguous())
    floor = physics["floor"]
    sim.add_surface_collider([1., 1., floor["height"]], [0., 0., 1.], surface=floor["surface"], friction=floor["friction"])
    sim.add_bounding_box()
    if physics["transfer"] == "guarded-flip":
        enable_guarded_flip(sim)
    tensor = torch.eye(3, device="cuda").repeat(len(initial), 1, 1)
    return SimulationRuntime(copy.deepcopy(config), sim, tensor, wp.from_torch(tensor, dtype=wp.mat33),
                             wp.zeros(len(initial), dtype=int, device="cuda:0"))
