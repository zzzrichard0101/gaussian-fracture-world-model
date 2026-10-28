"""Vectorized copy of the reduced fixture's Verlet update, checked against it.

Initial states must come from valid history replay. This accelerates synthetic
data generation only; it does not change the material/contact law.
"""
import numpy as np

from wmfracture.cohesive import InterfaceLaw
from wmfracture.cohesive_inference import tick


def batch_rollout(model, states, probes, impacts, dt, sample_times, probe_time=.004, future_time=.036, start_elapsed=0., fracture_energies=None):
    law = InterfaceLaw(**model["law"])
    n = len(states)
    if fracture_energies is None:
        final_opening = law.failure_opening
    else:
        energies = np.asarray(fracture_energies, dtype=float)
        if energies.shape != (n,) or not np.isfinite(energies).all() or (energies <= 0).any():
            raise ValueError("Invalid per-candidate fracture energies")
        final_opening = 2*energies/law.peak_traction
        if (final_opening <= law.initiation_opening).any():
            raise ValueError("Candidate failure opening precedes damage initiation")
    probes, impacts = np.asarray(probes, dtype=float), np.asarray(impacts, dtype=float)
    if probes.shape != (n,) or impacts.shape != (n,) or not n or not np.isfinite([probes, impacts]).all():
        raise ValueError("Invalid batch actions")
    ml, mr, width = model["mass_left"], model["mass_right"], model["width"]
    omega = np.sqrt(law.area * max(law.tensile_stiffness, law.compressive_stiffness) * (1/ml+1/mr))
    if not 0 < dt * omega < 1:
        raise ValueError("Unstable batch dt")
    if sample_times != sorted(set(sample_times)):
        raise ValueError("Sample clocks must be ordered and unique")
    wanted = {tick(value, dt): index for index, value in enumerate(sample_times)}
    probe_tick, future_tick = tick(probe_time, dt), tick(future_time, dt)
    first = tick(start_elapsed, dt)
    if not wanted or min(wanted) < first:
        raise ValueError("Invalid sampling horizon")
    xleft, xright, vleft, vright, kappa = [np.array([getattr(state, key) for state in states], dtype=np.float64)
        for key in ("x_left", "x_right", "v_left", "v_right", "maximum_opening")]
    if (not np.isfinite([xleft, xright, vleft, vright, kappa]).all()
            or (kappa < np.maximum(xright-xleft-width, 0)-1e-12).any()):
        raise ValueError("Invalid release state")
    output = {key: np.empty((n, len(sample_times)), dtype=np.float64)
        for key in ("x_left", "x_right", "v_left", "v_right", "opening", "damage", "maximum_opening", "time", "external_work")}
    output["step_index"] = np.empty((n, len(sample_times)), dtype=np.int64)
    work = np.array([state.external_work for state in states], dtype=np.float64)
    initial_times = np.array([state.time for state in states])
    initial_steps = np.array([state.step_index for state in states], dtype=np.int64)
    def damage():
        safe = np.maximum(kappa, law.initiation_opening)
        return np.clip(final_opening * (safe-law.initiation_opening) /
                       (safe*(final_opening-law.initiation_opening)), 0, 1)
    def force():
        gap = xright-xleft-width
        return law.area * np.where(gap < 0, law.compressive_stiffness*gap,
                                   (1-damage())*law.tensile_stiffness*gap)
    for step in range(first, max(wanted)+1):
        if step in wanted:
            index = wanted[step]
            for key, value in [("x_left", xleft), ("x_right", xright), ("opening", xright-xleft-width),
                               ("damage", damage()), ("maximum_opening", kappa), ("v_left", vleft), ("v_right", vright),
                               ("time", initial_times+(step-first)*dt), ("step_index", initial_steps+step-first), ("external_work", work)]:
                output[key][:, index] = value
        # Observe before impulse, exactly as in known_action_rollout.
        if step == probe_tick:
            work += .5*ml*((vleft+probes)**2-vleft**2)
            vleft += probes
        if step == future_tick:
            work += .5*ml*((vleft+impacts)**2-vleft**2)
            vleft += impacts
        if step < max(wanted):
            acceleration = force()
            vleft += .5*dt*acceleration/ml
            vright -= .5*dt*acceleration/mr
            xleft += dt*vleft
            xright += dt*vright
            kappa = np.maximum(kappa, np.maximum(xright-xleft-width, 0))
            acceleration = force()
            vleft += .5*dt*acceleration/ml
            vright -= .5*dt*acceleration/mr
    if not all(np.isfinite(value).all() for value in output.values()):
        raise ValueError("Nonfinite batch physics")
    return output
