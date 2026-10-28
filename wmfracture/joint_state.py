"""Explicit material hypotheses and budgeted 2D subdivision; no truth inputs."""
import copy
import numpy as np


def material_model(known, energy):
    if "fracture_energy" in known["law"]:
        raise ValueError("Unknown fracture energy must be absent from known-model input")
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Invalid candidate material")
    model = copy.deepcopy(known)
    model["law"]["fracture_energy"] = float(energy)
    return model


def matched_history(known, energy, stiffness_ratio):
    onset = known["law"]["peak_traction"]/known["law"]["tensile_stiffness"]
    final = 2*energy/known["law"]["peak_traction"]
    if not 0 < stiffness_ratio < 1 or final <= onset:
        raise ValueError("Invalid matched-stiffness construction")
    return onset*final/(onset+stiffness_ratio*(final-onset))


def physical_parameters(unit, bounds):
    unit = np.asarray(unit, dtype=float)
    if unit.shape != (2,) or not np.isfinite(unit).all() or (unit < 0).any() or (unit > 1).any():
        raise ValueError("Candidate outside normalized parameter bounds")
    values = np.array(bounds)[:, 0]+unit*np.ptp(np.array(bounds), axis=1)
    return round(float(values[0]), 15), round(float(values[1]), 12)


def partition_search(objective, budget=49):
    """Two local cell splits then one widest split; partial final batch retained."""
    observed = {}
    trace = []
    def evaluate(point):
        point = tuple(round(float(x), 15) for x in point)
        if point not in observed:
            observed[point] = float(objective(np.array(point)))
    for x in (0., .5, 1.):
        for y in (0., .5, 1.):
            evaluate((x, y))
    leaves = [(x, x+.5, y, y+.5) for x in (0., .5) for y in (0., .5)]
    step = 0
    while len(observed) < budget:
        def corners(cell):
            a, b, c, d = cell
            return [(a, c), (a, d), (b, c), (b, d)]
        def merit(cell):
            return min(observed[tuple(round(v, 15) for v in point)] for point in corners(cell))
        def area(cell):
            return (cell[1]-cell[0])*(cell[3]-cell[2])
        global_step = step % 3 == 2
        cell = min(leaves, key=lambda c: (-area(c), merit(c), c) if global_step else (merit(c), -area(c), c))
        a, b, c, d = cell
        midx, midy = (a+b)/2, (c+d)/2
        proposals = [(midx, midy), (a, midy), (b, midy), (midx, c), (midx, d)]
        added = []
        for point in proposals:
            key = tuple(round(float(x), 15) for x in point)
            if key in observed:
                continue
            if len(observed) >= budget:
                break
            evaluate(point); added.append(key)
        trace.append({"step": step, "global_split": global_step, "cell": list(cell), "evaluated": added})
        if len(observed) < budget:
            leaves.remove(cell)
            leaves.extend([(a, midx, c, midy), (a, midx, midy, d), (midx, b, c, midy), (midx, b, midy, d)])
        step += 1
    return {"samples": [list(point) for point in observed], "scores": list(observed.values()),
        "callback_evaluations": len(observed), "distinct_locations": len(observed), "trace": trace}
