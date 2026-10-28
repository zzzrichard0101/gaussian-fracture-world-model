"""Declarative, frame-boundary external impulses, independent of constitutive state.

An event at frame k acts AFTER the image/checkpoint at k and BEFORE k -> k+1.
This is a prescribed external momentum impulse, not a resolved striker contact.
"""
import math

import torch
import warp as wp


def validate_schedule(schedule):
    if not isinstance(schedule, list):
        raise ValueError("loading_schedule must be a list")
    identifiers = set()
    last_frame = -1
    for event in schedule:
        if not isinstance(event, dict):
            raise ValueError("Impulse must be a JSON object")
        required = {"id", "frame", "kind", "region", "delta_velocity"}
        if event.get("region") == "sphere":
            required |= {"center", "radius"}
        if set(event) != required or event.get("kind") != "momentum_impulse":
            raise ValueError("Unsupported impulse schema/kind")
        identifier, frame = event["id"], event["frame"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Impulse IDs must be unique nonempty strings")
        if type(frame) is not int or frame < 0 or frame < last_frame:
            raise ValueError("Impulse frames must be nonnegative integers in ascending order")
        if event["region"] not in ("all", "sphere"):
            raise ValueError("Impulse region must be all or sphere")
        for name in ("delta_velocity", "center"):
            if name in event:
                value = event[name]
                if (not isinstance(value, list) or len(value) != 3
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
                    raise ValueError(f"{name} must contain three finite numbers")
        if event["region"] == "sphere":
            radius = event["radius"]
            if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
                raise ValueError("Impulse radius must be positive and finite")
        identifiers.add(identifier)
        last_frame = frame
    return schedule


def apply_impulse(sim, event):
    """Change v only; measure external momentum and kinetic work in demo units."""
    with torch.no_grad():
        x = sim.export_particle_x_to_torch()
        velocity = wp.to_torch(sim.mpm_state.particle_v)
        if event["region"] == "all":
            weights = torch.ones(len(x), device=x.device)
        else:
            center = torch.tensor(event["center"], device=x.device)
            r2 = ((x - center) ** 2).sum(dim=1) / event["radius"] ** 2
            weights = (1 - r2).clamp_min(0).square()
        affected = int((weights > 0).sum())
        if affected == 0:
            raise ValueError(f"Impulse {event['id']} misses all particles")
        delta = weights[:, None] * torch.tensor(event["delta_velocity"], device=x.device)
        mass = wp.to_torch(sim.mpm_state.particle_mass)
        momentum = (mass[:, None] * delta).sum(dim=0)
        work = (mass * (velocity * delta + .5 * delta.square()).sum(dim=1)).sum()
        velocity.add_(delta)
        return {"id": event["id"], "frame": event["frame"], "time": float(sim.time),
                "affected_particles": affected, "weight_sum": float(weights.sum()),
                "momentum_increment": momentum.cpu().tolist(), "kinetic_work": float(work)}


def validate_load_log(schedule, frame_index, log):
    expected = [event for event in schedule if event["frame"] < frame_index]
    if [(event["id"], event["frame"]) for event in expected] != [(row["id"], row["frame"]) for row in log]:
        raise ValueError("Applied impulse log does not match checkpoint frame/schedule")
