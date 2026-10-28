"""Versioned HDF5 frame-boundary checkpoints; no Python object deserialization."""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import uuid

import h5py
import numpy as np
import torch
import warp as wp
from mpm_solver_warp.mpm_solver_warp import (MPMStateStruct, MPMModelStruct, MPMtailoredStruct, Dirichlet_collider)
from wmfracture.runtime import build_runtime
from wmfracture.loading import validate_load_log

SCHEMA = 2
APPEARANCE = {"initial", "volumes", "covariance", "opacity", "colors", "binding", "offsets",
              "source_ids", "chosen", "beta", "reinforced"}
STRUCTS = {"state": ("mpm_state", MPMStateStruct), "model": ("mpm_model", MPMModelStruct),
           "tailored": ("tailored_struct_for_bc", MPMtailoredStruct)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def scalar(value):
    if isinstance(value, (bool, int, float)):
        return value
    if hasattr(value, "value"):
        return value.value
    return [float(item) for item in value]


def scalar_struct(instance, description):
    return {name: scalar(getattr(instance, name)) for name in description.vars
            if not isinstance(getattr(instance, name), wp.array)}


def physics_fingerprint():
    source, project = Path(os.environ["WM_SOURCE_ROOT"]), Path(os.environ["WM_PROJECT_ROOT"])
    paths = {"upstream/" + name: source / "mpm_solver_warp" / name
             for name in ("mpm_solver_warp.py", "mpm_utils.py", "warp_utils.py", "engine_utils.py")}
    paths.update({"local/" + name: project / "wmfracture" / name for name in ("runtime.py", "solver_adapter.py", "loading.py")})
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def versions():
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "warp": importlib.metadata.version("warp-lang"), "numpy": np.__version__}


def kernels(sim):
    return [{"module": item.module.name, "key": item.key} for item in sim.grid_postprocess]


def ensure_supported(runtime):
    sim = runtime.sim
    if getattr(sim, "_experimental_constitutive_override", None):
        raise ValueError("Checkpoint does not support experimental constitutive overrides")
    validate_load_log(runtime.config["physics"].get("loading_schedule", []), runtime.frame_index, runtime.applied_loads)
    if sim.pre_p2g_operations or sim.impulse_params or sim.particle_velocity_modifiers or sim.particle_velocity_modifier_params:
        raise ValueError("Checkpoint cannot serialize custom force/velocity callbacks")
    if any(item is not None for item in sim.modify_bc):
        raise ValueError("Checkpoint cannot serialize moving/custom boundary callbacks")
    expected = 3 if runtime.config["physics"]["transfer"] == "guarded-flip" else 2
    expected_keys = ["MPM_Simulator_WARP__add_surface_collider__locals__collide",
                     "MPM_Simulator_WARP__add_bounding_box__locals__collide"]
    if expected == 3:
        expected_keys.append("guard_empty_grid_mass")
    if len(sim.collider_params) != expected or [item["key"] for item in kernels(sim)] != expected_keys:
        raise ValueError("Unsupported boundary sequence; refusing incomplete checkpoint")
    expected_time = runtime.frame_index * runtime.config["physics"]["frame_dt"]
    if abs(sim.time - expected_time) > 1e-8:
        raise ValueError("Checkpoint must be at a complete frame boundary")


def put(group, name, value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif isinstance(value, wp.array):
        value = value.numpy()
    value = np.ascontiguousarray(value)
    if value.dtype.kind not in "bifu" or not np.isfinite(value).all():
        raise ValueError(f"Invalid checkpoint array: {group.name}/{name}")
    dataset = group.create_dataset(name, data=value, compression="gzip", compression_opts=1, shuffle=True)
    dataset.attrs["sha256"] = digest(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str, "sha256": dataset.attrs["sha256"]}


def save_checkpoint(path, runtime, appearance, provenance=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if set(appearance) != APPEARANCE:
        raise ValueError("Checkpoint appearance schema mismatch")
    ensure_supported(runtime)
    wp.synchronize()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    sim = runtime.sim
    metadata = {"schema": SCHEMA, "boundary": "frame-end", "config": runtime.config,
                "frame_index": runtime.frame_index, "global_step": runtime.global_step, "time": sim.time,
                "applied_loads": runtime.applied_loads,
                "n_particles": sim.n_particles, "versions": versions(), "physics_fingerprint": physics_fingerprint(),
                "device_recorded": torch.cuda.get_device_name(0), "provenance": provenance or {},
                "kernels": kernels(sim), "colliders": [scalar_struct(item, Dirichlet_collider) for item in sim.collider_params],
                "struct_scalars": {name: scalar_struct(getattr(sim, attribute), description)
                                   for name, (attribute, description) in STRUCTS.items()}, "datasets": {},
                "warp_rng": "No stochastic Warp kernels in the supported stepping pipeline."}
    try:
        with h5py.File(temporary, "x") as output:
            for group_name, (attribute, description) in STRUCTS.items():
                group = output.create_group(group_name)
                for name in description.vars:
                    value = getattr(sim, attribute).__getattribute__(name)
                    if isinstance(value, wp.array):
                        metadata["datasets"][group.name + "/" + name] = put(group, name, value)
            extra = output.create_group("runtime")
            for name, value in {"total_F": runtime.total_F_tensor, "invalid_F": runtime.invalid_F}.items():
                metadata["datasets"][extra.name + "/" + name] = put(extra, name, value)
            group = output.create_group("appearance")
            for name, value in appearance.items():
                metadata["datasets"][group.name + "/" + name] = put(group, name, value)
            rng = output.create_group("rng")
            values = {"torch_cpu": torch.get_rng_state()}
            for index, state in enumerate(torch.cuda.get_rng_state_all()):
                values[f"torch_cuda_{index}"] = state
            np_state = np.random.get_state()
            values["numpy_keys"] = np_state[1]
            metadata["numpy_rng"] = {"algorithm": np_state[0], "pos": np_state[2], "has_gauss": np_state[3], "cached_gaussian": np_state[4]}
            metadata["python_rng"] = random.getstate()
            for name, value in values.items():
                metadata["datasets"][rng.name + "/" + name] = put(rng, name, value)
            payload = canonical(metadata)
            output.attrs["metadata"] = payload
            output.attrs["metadata_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
            output.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return metadata


@dataclass
class LoadedCheckpoint:
    runtime: object
    appearance: dict
    metadata: dict
    verified_arrays: int


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def load_checkpoint(path):
    wp.init()
    with h5py.File(path, "r") as input_file:
        payload = input_file.attrs["metadata"]
        if hashlib.sha256(payload.encode()).hexdigest() != input_file.attrs["metadata_sha256"]:
            raise ValueError("Checkpoint metadata checksum mismatch")
        metadata = json.loads(payload)
        if metadata["schema"] != SCHEMA or metadata["boundary"] != "frame-end":
            raise ValueError("Unsupported checkpoint format/boundary")
        if metadata["versions"] != versions() or metadata["physics_fingerprint"] != physics_fingerprint():
            raise ValueError("Checkpoint physics code or runtime version mismatch")
        datasets = {}
        input_file.visititems(lambda name, value: datasets.update({"/" + name: value}) if isinstance(value, h5py.Dataset) else None)
        if set(datasets) != set(metadata["datasets"]):
            raise ValueError("Checkpoint dataset schema mismatch")
        arrays = {}
        for name, dataset in datasets.items():
            array = dataset[:]
            expected = metadata["datasets"][name]
            if list(array.shape) != expected["shape"] or array.dtype.str != expected["dtype"] or digest(array) != expected["sha256"]:
                raise ValueError(f"Checkpoint dataset integrity mismatch: {name}")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite checkpoint data: {name}")
            arrays[name] = array
    if {name.split("/")[-1] for name in arrays if name.startswith("/appearance/")} != APPEARANCE:
        raise ValueError("Incomplete appearance state")
    appearance = {name: torch.from_numpy(arrays["/appearance/" + name]).cuda() for name in APPEARANCE}
    runtime = build_runtime(metadata["config"], appearance["initial"], appearance["volumes"], appearance["beta"])
    sim = runtime.sim
    if sim.n_particles != metadata["n_particles"] or kernels(sim) != metadata["kernels"]:
        raise ValueError("Checkpoint particle/boundary reconstruction mismatch")
    verified = 0
    for group_name, (attribute, description) in STRUCTS.items():
        instance = getattr(sim, attribute)
        for name in description.vars:
            target = getattr(instance, name)
            key = "/" + group_name + "/" + name
            if isinstance(target, wp.array):
                if key not in arrays or tuple(target.numpy().shape) != arrays[key].shape:
                    raise ValueError(f"Checkpoint field shape mismatch: {key}")
                target.assign(arrays[key])
                if not np.array_equal(target.numpy(), arrays[key]):
                    raise RuntimeError(f"GPU restore differs: {key}")
                verified += 1
            else:
                if name not in metadata["struct_scalars"][group_name]:
                    raise ValueError(f"Missing scalar: {group_name}/{name}")
                value = metadata["struct_scalars"][group_name][name]
                setattr(instance, name, type(target)(*value) if isinstance(value, list) else value)
        if scalar_struct(instance, description) != metadata["struct_scalars"][group_name]:
            raise RuntimeError(f"Scalar restore mismatch: {group_name}")
    for target, values in zip(sim.collider_params, metadata["colliders"]):
        for name in Dirichlet_collider.vars:
            value = values[name]
            previous = getattr(target, name)
            setattr(target, name, type(previous)(*value) if isinstance(value, list) else value)
        if scalar_struct(target, Dirichlet_collider) != values:
            raise RuntimeError("Collider restore mismatch")
    runtime.total_F_tensor.copy_(torch.from_numpy(arrays["/runtime/total_F"]).cuda())
    runtime.invalid_F.assign(arrays["/runtime/invalid_F"])
    for name, tensor in [("total_F", runtime.total_F_tensor), ("invalid_F", wp.to_torch(runtime.invalid_F))]:
        if not np.array_equal(tensor.cpu().numpy(), arrays["/runtime/" + name]):
            raise RuntimeError(f"Runtime restore differs: {name}")
        verified += 1
    for name, tensor in appearance.items():
        if not np.array_equal(tensor.cpu().numpy(), arrays["/appearance/" + name]):
            raise RuntimeError(f"Appearance restore differs: {name}")
        verified += 1
    sim.time = metadata["time"]
    runtime.frame_index, runtime.global_step = metadata["frame_index"], metadata["global_step"]
    runtime.applied_loads = metadata["applied_loads"]
    ensure_supported(runtime)
    torch.set_rng_state(torch.from_numpy(arrays["/rng/torch_cpu"]))
    cuda_states = [arrays[name] for name in sorted(arrays) if name.startswith("/rng/torch_cuda_")]
    if len(cuda_states) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA RNG device count mismatch")
    torch.cuda.set_rng_state_all([torch.from_numpy(item) for item in cuda_states])
    rng = metadata["numpy_rng"]
    np.random.set_state((rng["algorithm"], arrays["/rng/numpy_keys"], rng["pos"], rng["has_gauss"], rng["cached_gaussian"]))
    random.setstate(_tuples(metadata["python_rng"]))
    return LoadedCheckpoint(runtime, appearance, metadata, verified)
