"""Numerical adapter for the pinned solver; upstream files remain unchanged."""
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPMStateStruct, MPMModelStruct, Dirichlet_collider


@wp.kernel
def guard_empty_grid_mass(time: float, dt: float, state: MPMStateStruct,
                          model: MPMModelStruct, param: Dirichlet_collider):
    i, j, k = wp.tid()
    # Upstream g2p_flip evaluates weighted old momentum / mass even when the
    # interpolation weight and grid mass are both exactly zero. A finite dummy
    # denominator only on empty nodes makes that contribution zero. Occupied
    # node masses/momenta and collision velocities are not changed. zero_grid
    # clears the dummy masses at the beginning of the next step.
    if state.grid_m[i, j, k] == 0.0:
        state.grid_m[i, j, k] = 1.0e-30


def enable_guarded_flip(sim):
    if getattr(sim, "_empty_mass_guard_installed", False):
        return
    sim.grid_postprocess.append(guard_empty_grid_mass)
    sim.collider_params.append(Dirichlet_collider())
    sim.modify_bc.append(None)
    sim._empty_mass_guard_installed = True


@wp.kernel
def accumulate_total_deformation(state: MPMStateStruct, total: wp.array(dtype=wp.mat33),
                                  invalid: wp.array(dtype=int)):
    p = wp.tid()
    # In upstream g2p, F_trial(new) = (I + dt grad(v)) F_elastic(after return).
    # Recover the kinematic increment before multiplying the accumulated map.
    # F_elastic itself is reset by plastic return and is not a geometric history.
    elastic = state.particle_F[p]
    if wp.abs(wp.determinant(elastic)) < 1.0e-8:
        invalid[p] = 1
    else:
        increment = state.particle_F_trial[p] * wp.inverse(elastic)
        total[p] = increment * total[p]
