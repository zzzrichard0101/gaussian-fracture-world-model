import numpy as np
import pytest
import torch
import warp as wp

from wmfracture.loading import apply_impulse, validate_schedule
from wmfracture.checkpoint import save_checkpoint, load_checkpoint
from test_checkpoint import small_bundle


def event(identifier="load", frame=1, region="all"):
    result = {"id": identifier, "frame": frame, "kind": "momentum_impulse", "region": region,
              "delta_velocity": [.2, -.1, -.3]}
    if region == "sphere":
        result.update(center=[.35, .35, .35], radius=.12)
    return result


@pytest.mark.parametrize("schedule", [
    [event(), event()], [event(frame=-1)], [event(frame=True)],
    [dict(event(), delta_velocity=[float("nan"), 0, 0])],
    [dict(event(region="sphere"), radius=0)],
    [event(frame=2), event("early", frame=1)],
])
def test_rejects_ambiguous_or_invalid_schedule(schedule):
    with pytest.raises(ValueError):
        validate_schedule(schedule)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA impulse conservation")
def test_external_impulse_conserves_unmodified_state_and_accounts_for_work():
    runtime, _ = small_bundle()
    runtime.advance_frame()
    state = runtime.sim.mpm_state
    before = {name: getattr(state, name).numpy().copy() for name in
              ("particle_x", "particle_F", "particle_F_trial", "particle_C", "particle_Jp")}
    mass = wp.to_torch(state.particle_mass).clone()
    velocity = wp.to_torch(state.particle_v).clone()
    old_momentum = (mass[:, None] * velocity).sum(0)
    old_energy = (.5 * mass * velocity.square().sum(1)).sum()
    audit = apply_impulse(runtime.sim, event(region="sphere"))
    new_velocity = wp.to_torch(state.particle_v)
    torch.testing.assert_close((mass[:, None] * new_velocity).sum(0) - old_momentum,
                               torch.tensor(audit["momentum_increment"], device="cuda"), atol=1e-8, rtol=1e-5)
    assert float((.5 * mass * new_velocity.square().sum(1)).sum() - old_energy) == pytest.approx(audit["kinetic_work"], abs=1e-8)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(state, name).numpy(), value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scheduled restart")
def test_restart_before_and_after_load_executes_future_impulse_once(tmp_path):
    runtime, appearance = small_bundle()
    runtime.config["physics"]["loading_schedule"] = [event(), event("future", frame=2)]
    runtime.advance_frame()
    save_checkpoint(tmp_path / "before.h5", runtime, appearance)
    resumed = load_checkpoint(tmp_path / "before.h5")
    assert resumed.runtime.applied_loads == []
    runtime.advance_frame()
    resumed.runtime.advance_frame()
    assert [row["id"] for row in resumed.runtime.applied_loads] == ["load"]
    save_checkpoint(tmp_path / "after.h5", resumed.runtime, resumed.appearance)
    restored = load_checkpoint(tmp_path / "after.h5")
    runtime.advance_frame()
    restored.runtime.advance_frame()
    assert [row["id"] for row in restored.runtime.applied_loads] == ["load", "future"]
    np.testing.assert_allclose(restored.runtime.sim.export_particle_v_to_torch().cpu().numpy(),
                               runtime.sim.export_particle_v_to_torch().cpu().numpy(), atol=2e-4, rtol=2e-4)
