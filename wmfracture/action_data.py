"""Narrow input contracts and observation-only features for the adapter probe."""
import numpy as np


INPUT_KEYS = {"rgbd_features", "prefix_index", "actions", "accepted", "rgbd_best", "bank_index"}


def validate_inputs(arrays):
    if set(arrays) != INPUT_KEYS:
        raise ValueError("Unexpected action-model inputs, potentially privileged targets")
    n = len(arrays["actions"])
    if arrays["actions"].shape != (n, 2) or arrays["rgbd_features"].shape != (n, 45):
        raise ValueError("Wrong observation/action dimensions")
    if (arrays["accepted"].shape != (n, 13)
        or any(arrays[key].shape != (n,) for key in ("prefix_index", "rgbd_best", "bank_index"))
        or not all(np.isfinite(value).all() for value in arrays.values())):
        raise ValueError("Malformed/nonfinite prediction inputs")
    return arrays


def assert_group_split(groups):
    for first, values in groups.items():
        for second, other in groups.items():
            if first != second and set(values) & set(other):
                raise ValueError("History-group leakage across splits")


def prefix_features(rgb, depth, alpha):
    """Fixed blue/orange part summaries; no latent physical labels or future."""
    features = []
    yy, xx = np.mgrid[:rgb.shape[1], :rgb.shape[2]]
    for color, z, opacity in zip(rgb, depth, alpha):
        masks = [(color[..., 2]-color[..., 0] > .10) & (color[..., 2]-color[..., 1] > .05),
                 (color[..., 0]-color[..., 2] > .10) & (color[..., 0]-color[..., 1] > .15)]
        for mask in masks:
            mask &= opacity > .5
            if mask.sum() < 20:
                raise ValueError("Known colored part not visible in observed prefix")
            weights = opacity[mask]
            features.extend([np.average(xx[mask]/rgb.shape[2], weights=weights),
                np.average(yy[mask]/rgb.shape[1], weights=weights),
                np.average(z[mask], weights=weights), mask.mean()])
    result = np.array(features, dtype=np.float32)
    if result.shape != (32,) or not np.isfinite(result).all():
        raise ValueError("Prefix must contain four valid RGB-D frames")
    return result


def project_to_bank(predicted_gaps, accepted, bank):
    """Choose a reachable candidate; never expand an empty observation match."""
    indices = np.flatnonzero(accepted)
    if not len(indices):
        return -1
    if not np.isfinite(predicted_gaps).all() or not np.isfinite(bank).all():
        raise ValueError("Nonfinite dynamics projection")
    return int(indices[np.argmin(np.mean((bank[indices]-predicted_gaps)**2, axis=1))])
