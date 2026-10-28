from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch
import warp as wp

from wmfracture.experimental_nacc import ray_pressure, compute_stress_corrected
from mpm_utils import compute_stress_from_F_trial
from wmfracture.runtime import build_runtime
from wmfracture.checkpoint import save_checkpoint
from test_checkpoint import small_bundle


@wp.kernel
def evaluate_ray(values: wp.array(dtype=float), output: wp.array(dtype=float)):
    output[0] = ray_pressure(values[0], values[1], values[2], values[3], values[4])


def decimal_intersection(values):
    # Independent high precision center/radius construction; direct subtraction
    # is safe with 70 digits and does not use the implementation's y correction.
    with localcontext() as context:
        context.prec = 70
        p, q, p0, beta, M = map(lambda x: Decimal(str(x)), values)
        center = (1 - beta) * p0 / 2
        half = (1 + beta) * p0 / 2
        dp = p - center
        radius = (M * M * half * half / (M * M * dp * dp + (1 + 2 * beta) * q * q)).sqrt()
        return float(center + radius * dp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA constitutive regression")
@pytest.mark.parametrize("beta", [1., 10., 3e9])
def test_scaled_ray_matches_high_precision_geometric_intersection(beta):
    wp.init()
    values = np.array([314.4354, 184.0863, 334.1617, beta, 1.8503437], dtype=np.float32)
    output = wp.zeros(1, dtype=float, device="cuda:0")
    wp.launch(evaluate_ray, dim=1, inputs=[wp.array(values, device="cuda:0"), output], device="cuda:0")
    assert float(output.numpy()[0]) == pytest.approx(decimal_intersection(values), rel=2e-5, abs=.002)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA captured-state regression")
@pytest.mark.parametrize("captured,initial_jp,expected_jp", [
    ([[[1.0008510351, .0002299067, -.0001021929],
       [.0002262099, 1.0018239021, -.0000806110],
       [.0006401886, .0007319683, .8772011399]]], -.04, -.05805756),
    ([[[.8980616331, .0018436267, .0439527817],
       [.0121449903, 1.0246757269, -.0400664546],
       [.0171879120, .0289661884, .6727076173]]], -.1192613095, -.1192613),
])
def test_captured_finite_state_original_overflows_corrected_state_is_finite(captured, initial_jp, expected_jp):
    wp.init()
    config = {"physics": {"grid": 16, "domain_size": 2., "material": {"material": "watermelon",
        "E": 2000., "nu": .38, "density": 1., "beta": 3e9, "xi": 3., "hardening": 1.,
        "alpha_0": -.04, "g": [0., 0., 0.]}, "initial_velocity": [0., 0., 0.],
        "floor": {"height": .1, "surface": "sticky", "friction": 0.}, "transfer": "guarded-flip"},
        "render": {"deformation": "total"}}
    runtime = build_runtime(config, torch.tensor([[1., 1., 1.]], device="cuda"),
                            torch.tensor([.001], device="cuda"), np.array([3e9], dtype=np.float32))
    state, model = runtime.sim.mpm_state, runtime.sim.mpm_model
    state.particle_F_trial.assign(np.array(captured, dtype=np.float32))
    state.particle_Jp.assign(np.array([initial_jp], dtype=np.float32))
    wp.launch(compute_stress_from_F_trial, dim=1, inputs=[state, model, .0001], device="cuda:0")
    assert not np.isfinite(state.particle_Jp.numpy()).all()
    projected = state.particle_F.numpy().copy()
    state.particle_Jp.assign(np.array([initial_jp], dtype=np.float32))
    wp.launch(compute_stress_corrected, dim=1, inputs=[state, model, .0001], device="cuda:0")
    assert np.isfinite(state.particle_Jp.numpy()).all()
    assert float(state.particle_Jp.numpy()[0]) == pytest.approx(expected_jp, abs=1e-4)
    np.testing.assert_array_equal(projected, state.particle_F.numpy())
    assert np.isfinite(state.particle_stress.numpy()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint override guard")
def test_experimental_physics_cannot_be_saved_as_supported_checkpoint(tmp_path):
    runtime, appearance = small_bundle()
    runtime.sim._experimental_constitutive_override = "scaled-nacc-ray-hardening-v1"
    with pytest.raises(ValueError, match="experimental constitutive"):
        save_checkpoint(tmp_path / "unsupported.h5", runtime, appearance)
