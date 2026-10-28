"""Compress fixed, observation-matched physical hypotheses without truth labels.

Farthest-first traversal is a conventional k-center heuristic, not a new solver.
"""
import numpy as np


def response_features(gaps, failure, gap_scale_m):
    gaps, failure = np.asarray(gaps, float), np.asarray(failure, float)
    if gaps.ndim != 3 or failure.shape != gaps.shape[:2] or not np.isfinite(gaps).all() or gap_scale_m <= 0:
        raise ValueError("Expected candidate/action/time curves and candidate/action events")
    n, actions, times = gaps.shape
    return np.concatenate((gaps.reshape(n, -1)/(gap_scale_m*np.sqrt(actions*times)),
                           failure/np.sqrt(actions)), axis=1)


def select_hypotheses(parameters, scores, accepted, count, rule, bounds, features=None, seed=0):
    parameters, scores = np.asarray(parameters, float), np.asarray(scores, float)
    accepted = list(accepted)
    if (parameters.ndim != 2 or parameters.shape[1] != 2 or scores.shape != (len(parameters),)
        or not np.isfinite(parameters).all() or not np.isfinite(scores).all()
        or len(set(accepted)) != len(accepted) or any(i < 0 or i >= len(parameters) for i in accepted)
        or count < 1):
        raise ValueError("Invalid physical candidate set or count")
    if not accepted:
        return []
    ranked = sorted(accepted, key=lambda i: (scores[i], tuple(parameters[i])))
    if rule == "all":
        return ranked
    count = min(count, len(ranked))
    if rule == "residual":
        return ranked[:count]
    if rule == "random":
        rng = np.random.default_rng(seed)
        return ranked[:1]+rng.choice(ranked[1:], count-1, replace=False).tolist()
    if rule == "parameter":
        bounds = np.asarray(bounds, float)
        features = (parameters-bounds[:, 0])/np.ptp(bounds, axis=1)
    elif rule != "response":
        raise ValueError("Unknown compression rule")
    features = np.asarray(features, float)
    if features.ndim != 2 or len(features) != len(parameters) or not np.isfinite(features).all():
        raise ValueError("Invalid hypothesis features")
    selected = ranked[:1]
    distance = np.sum((features-features[selected[0]])**2, axis=1)
    while len(selected) < count:
        next_index = min((i for i in ranked if i not in selected), key=lambda i: (-distance[i], scores[i], tuple(parameters[i])))
        selected.append(next_index)
        distance = np.minimum(distance, np.sum((features-features[next_index])**2, axis=1))
    return selected
