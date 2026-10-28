"""Heuristic task-weighted experiment selection with finite-set damage checks.

Not a calibrated information gain estimator or a safety guarantee.
"""
import numpy as np


def rgbd_summary(frame):
    rgb, depth, alpha = frame["rgb"], frame["depth"], frame["object_alpha"]
    y, x = np.indices(depth.shape)
    masks = [(rgb[..., 2] > 1.2*rgb[..., 0]) & (rgb[..., 2] > 1.1*rgb[..., 1]),
             (rgb[..., 0] > 1.4*rgb[..., 2]) & (rgb[..., 0] > 1.2*rgb[..., 1])]
    output = []
    for mask in masks:
        mask = mask & (alpha > .5)
        if not mask.any():
            raise ValueError("Known colored block is not visible")
        output.extend([float(x[mask].mean()/depth.shape[1]), float(y[mask].mean()/depth.shape[0]),
                       float(depth[mask].mean()), float(mask.mean())])
    return np.array(output)


def task_weights(design_gaps, design_failure, scale_m):
    gaps, events = np.asarray(design_gaps, float), np.asarray(design_failure, float)
    if gaps.ndim != 3 or events.shape != gaps.shape[:2] or not np.isfinite(gaps).all() or scale_m <= 0:
        raise ValueError("Invalid candidate design responses")
    distance = np.sqrt(np.mean((gaps[:, None]-gaps[None, :])**2, axis=(2, 3)))
    event_distance = np.mean(events[:, None] != events[None, :], axis=2)
    return np.maximum(np.minimum(distance/scale_m, 1.), event_distance)


def select_probe(actions, summaries, end_damage, policy, feature_scales, damage_cap, weights=None, seed=0):
    actions = np.asarray(actions, float)
    summaries, end_damage = np.asarray(summaries, float), np.asarray(end_damage, float)
    if (actions.ndim != 1 or len(set(actions.tolist())) != len(actions) or 0. not in actions
        or summaries.ndim != 4 or summaries.shape[1] != len(actions)
        or end_damage.shape != summaries.shape[:2] or not np.isfinite(summaries).all()
        or not np.isfinite(end_damage).all() or (end_damage < 0).any() or (end_damage > 1).any()):
        raise ValueError("Invalid candidate probe predictions")
    n, count, _, dimensions = summaries.shape
    scales = np.asarray(feature_scales, float)
    if scales.shape != (dimensions,) or (scales <= 0).any() or not np.isfinite(scales).all() or damage_cap < 0:
        raise ValueError("Invalid experiment scales or risk cap")
    zero = int(np.where(actions == 0.)[0][0])
    pair = np.triu_indices(n, 1)
    spread, weighted = [], []
    if weights is None:
        weights = np.ones((n, n))
    weights = np.asarray(weights, float)
    if weights.shape != (n, n) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid task pair weights")
    pair_weights = weights[pair]
    for a in range(count):
        difference = (summaries[:, None, a]-summaries[None, :, a])/scales
        distinguishability = -np.expm1(-np.mean(difference**2, axis=(2, 3))/2)
        spread.append(float(distinguishability[pair].mean()) if len(pair[0]) else 0.)
        weighted.append(float(np.sum(distinguishability[pair]*pair_weights)/pair_weights.sum()) if pair_weights.sum() else 0.)
    spread, weighted = np.array(spread), np.array(weighted)
    added_damage = np.maximum(end_damage-end_damage[:, zero:zero+1], 0).max(axis=0)
    added_failure = ((end_damage[:, zero:zero+1] < 1.) & (end_damage == 1.)).any(axis=0)
    eligible = (added_damage <= damage_cap+1e-12) & ~added_failure
    eligible[zero] = True
    utility = weighted-weighted[zero]
    if policy == "none":
        selected = zero
    elif policy == "weak":
        selected = int(np.where(actions == .04)[0][0])
    elif policy == "strong":
        selected = int(np.where(actions == .20)[0][0])
    elif policy == "random":
        selected = int(np.random.default_rng(seed).integers(count))
    elif policy == "disagreement":
        selected = min(range(count), key=lambda i: (-spread[i], actions[i]))
    elif policy == "guarded":
        selected = min(np.flatnonzero(eligible).tolist(), key=lambda i: (-utility[i], actions[i]))
        if utility[selected] <= 1e-8:
            selected = zero
    else:
        raise ValueError("Unknown probing policy")
    return {"selected_action_index": selected, "selected_impulse": float(actions[selected]),
        "observable_pair_separation": spread.tolist(), "task_weighted_pair_separation": weighted.tolist(),
        "task_gain_over_passive": utility.tolist(), "max_candidate_added_damage": added_damage.tolist(),
        "candidate_new_separation": added_failure.tolist(), "eligible_actions": eligible.tolist(),
        "finite_hypothesis_guard_only": True}
