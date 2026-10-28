import pytest
import torch
import warp as wp

from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from wmfracture.solver_adapter import enable_guarded_flip, accumulate_total_deformation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration test")
def test_guarded_flip_fixes_exact_zero_weight_support_node():
    wp.init()

    def make_sim():
        # At x/dx=8.5, one quadratic interpolation weight is exactly zero.
        x = torch.full((1, 3), 8.5 / 16, device="cuda")
        sim = MPM_Simulator_WARP(1, n_grid=16)
        sim.load_initial_data_from_torch(x, torch.full((1,), .001, device="cuda"), n_grid=16)
        sim.set_parameters_dict({"material": "jelly", "E": 10., "nu": .2, "density": 1000., "g": [0., 0., -9.81]})
        sim.finalize_mu_lam()
        return sim

    unguarded = make_sim()
    unguarded.p2g2p(0, .0001, flip_pic=True, flip_pic_ratio=.7)
    assert not torch.isfinite(unguarded.export_particle_v_to_torch()).all(), "The pinned upstream reproduction changed"
    guarded = make_sim()
    enable_guarded_flip(guarded)
    for step in range(100):
        guarded.p2g2p(step, .0001, flip_pic=True, flip_pic_ratio=.7)
    x = guarded.export_particle_x_to_torch()
    v = guarded.export_particle_v_to_torch()
    assert torch.isfinite(x).all() and torch.isfinite(v).all()
    torch.testing.assert_close(v[0], torch.tensor([0., 0., -.0981], device="cuda"), atol=2e-5, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration test")
def test_geometric_history_survives_elastic_return_reset():
    wp.init()
    sim = MPM_Simulator_WARP(1, n_grid=16)
    total_tensor = torch.eye(3, device="cuda").reshape(1, 3, 3)
    total = wp.from_torch(total_tensor, dtype=wp.mat33)
    invalid = wp.zeros(1, dtype=int, device="cuda:0")
    # Two deformation increments with an intervening plastic/elastic reset.
    for elastic_diagonal, increment_diagonal in [([1.1, 1., 1.], [1.2, .9, 1.]), ([1.02, 1.01, 1.], [1.1, 1., 1.])]:
        elastic = torch.diag(torch.tensor(elastic_diagonal, device="cuda")).reshape(1, 3, 3)
        increment = torch.diag(torch.tensor(increment_diagonal, device="cuda")).reshape(1, 3, 3)
        sim.mpm_state.particle_F.assign(elastic.cpu().numpy())
        sim.mpm_state.particle_F_trial.assign((increment @ elastic).cpu().numpy())
        wp.launch(accumulate_total_deformation, dim=1, inputs=[sim.mpm_state, total, invalid], device="cuda:0")
    expected = torch.diag(torch.tensor([1.32, .9, 1.], device="cuda")).reshape(1, 3, 3)
    torch.testing.assert_close(total_tensor, expected)
    assert not wp.to_torch(invalid).any()
