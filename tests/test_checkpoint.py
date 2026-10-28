import random
import h5py
import numpy as np
import pytest
import torch
import warp as wp

from wmfracture.runtime import build_runtime
from wmfracture.checkpoint import load_checkpoint, save_checkpoint


def small_bundle(transfer="guarded-flip"):
    wp.init()
    config = {"physics": {"grid": 16, "domain_size": 1., "material": {"material": "jelly", "E": 100., "nu": .2,
               "density": 1000., "beta": 1., "g": [0., 0., -1.]}, "initial_velocity": [0., 0., -.1],
               "floor": {"height": .1, "surface": "sticky", "friction": 0.}, "transfer": "guarded-flip",
               "frame_dt": .005, "max_substep_dt": .001, "cfl": .3, "frames": 3},
              "render": {"deformation": "total"}}
    config["physics"]["transfer"] = transfer
    axis = torch.tensor([.35, .40], device="cuda")
    initial = torch.cartesian_prod(axis, axis, axis).contiguous()
    n = len(initial)
    volumes = torch.full((n,), .0001, device="cuda")
    beta = torch.linspace(.5, 1.2, n, device="cuda")
    runtime = build_runtime(config, initial, volumes, beta)
    appearance = {"initial": initial, "volumes": volumes, "covariance": torch.eye(3, device="cuda").repeat(n, 1, 1) * .0001,
                  "opacity": torch.full((n, 1), .9, device="cuda"), "colors": torch.ones((n, 3), device="cuda"),
                  "binding": torch.arange(n, device="cuda"), "offsets": torch.zeros_like(initial),
                  "source_ids": torch.arange(n, device="cuda"), "chosen": torch.arange(n, device="cuda"),
                  "beta": beta, "reinforced": torch.zeros(n, dtype=torch.bool, device="cuda")}
    return runtime, appearance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint integration")
@pytest.mark.parametrize("transfer", ["apic", "guarded-flip"])
def test_roundtrip_preserves_all_arrays_scalars_clocks_and_rng(tmp_path, transfer):
    runtime, appearance = small_bundle(transfer)
    runtime.advance_frame()
    # Preserve a spatial parameter field and a non-config scalar, not defaults.
    runtime.sim.mpm_model.softening = .27
    runtime.sim.mpm_state.particle_Jp.assign(np.linspace(-.05, -.03, len(appearance["initial"]), dtype=np.float32))
    path = tmp_path / "state.h5"
    metadata = save_checkpoint(path, runtime, appearance)
    expected = (random.random(), np.random.rand(4), torch.rand(4), torch.rand(4, device="cuda"))
    restored = load_checkpoint(path)
    actual = (random.random(), np.random.rand(4), torch.rand(4), torch.rand(4, device="cuda"))
    assert restored.runtime.sim.time == runtime.sim.time
    assert restored.runtime.global_step == runtime.global_step
    assert restored.runtime.frame_index == runtime.frame_index
    assert restored.runtime.sim.mpm_model.softening == .27
    assert restored.verified_arrays >= 30
    assert expected[0] == actual[0]
    np.testing.assert_array_equal(expected[1], actual[1])
    torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)
    torch.testing.assert_close(expected[3], actual[3], atol=0, rtol=0)
    assert metadata["boundary"] == "frame-end"
    with pytest.raises(FileExistsError):
        save_checkpoint(path, runtime, appearance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint integration")
@pytest.mark.parametrize("corruption", ["missing", "changed"])
def test_rejects_missing_or_corrupted_physics_state(tmp_path, corruption):
    runtime, appearance = small_bundle()
    path = tmp_path / "bad.h5"
    save_checkpoint(path, runtime, appearance)
    with h5py.File(path, "r+") as output:
        if corruption == "missing":
            del output["state/particle_Jp"]
        else:
            output["state/particle_x"][0, 0] += .1
    with pytest.raises(ValueError, match="schema|integrity"):
        load_checkpoint(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint integration")
def test_rejects_midframe_and_unknown_callbacks(tmp_path):
    runtime, appearance = small_bundle()
    runtime.sim.time = .001
    with pytest.raises(ValueError, match="frame boundary"):
        save_checkpoint(tmp_path / "midframe.h5", runtime, appearance)
    runtime.sim.time = 0.
    runtime.sim.modify_bc[0] = lambda *args: None
    with pytest.raises(ValueError, match="callbacks"):
        save_checkpoint(tmp_path / "callback.h5", runtime, appearance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint integration")
def test_rejects_unknown_checkpoint_version(tmp_path):
    import hashlib
    import json
    runtime, appearance = small_bundle()
    path = tmp_path / "future.h5"
    save_checkpoint(path, runtime, appearance)
    with h5py.File(path, "r+") as output:
        metadata = json.loads(output.attrs["metadata"])
        metadata["schema"] = 999
        payload = json.dumps(metadata)
        output.attrs["metadata"] = payload
        output.attrs["metadata_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        load_checkpoint(path)
