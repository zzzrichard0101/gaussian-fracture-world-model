"""Observation-only hypothesis weights and finite-mixture diagnostics."""
import numpy as np


EVIDENCE_KEYS = {"prefix_index", "scores", "accepted", "rgbd_features", "probe"}


def validate_evidence(arrays):
    if set(arrays) != EVIDENCE_KEYS:
        raise ValueError("Belief input must exclude future plans and privileged targets")
    n = len(arrays["prefix_index"])
    if (arrays["prefix_index"].shape != (n,) or arrays["probe"].shape != (n, 1)
        or arrays["scores"].shape != (n, 13) or arrays["accepted"].shape != (n, 13)
        or arrays["rgbd_features"].shape != (n, 45)
        or not all(np.isfinite(value).all() for value in arrays.values())
        or (arrays["scores"] < 0).any()):
        raise ValueError("Invalid observed belief evidence")
    return arrays


def state_targets(peaks, candidates):
    """Barycentric grid targets, not a continuous posterior density."""
    candidates = np.asarray(candidates, dtype=float)
    peaks = np.asarray(peaks, dtype=float)
    if (np.diff(candidates) <= 0).any() or not np.isfinite(peaks).all() or (peaks < candidates[0]).any() or (peaks > candidates[-1]).any():
        raise ValueError("History targets outside ordered candidate support")
    result = np.zeros((len(peaks), len(candidates)))
    for row, peak in enumerate(peaks):
        high = int(np.searchsorted(candidates, peak, side="right"))
        if high == len(candidates):
            result[row, -1] = 1
        else:
            low = max(0, high-1)
            if high == low:
                result[row, low] = 1
            else:
                fraction = (peak-candidates[low])/(candidates[high]-candidates[low])
                result[row, low], result[row, high] = 1-fraction, fraction
    return result


def energy_weights(scores, accepted, prior, temperature, correction=None):
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Invalid observation energy temperature")
    scores, accepted, prior = np.asarray(scores), np.asarray(accepted, dtype=bool), np.asarray(prior)
    if not np.isfinite(scores).all() or (prior <= 0).any() or not np.isclose(prior.sum(), 1):
        raise ValueError("Invalid energy/prior")
    logits = np.log(prior)[None]-(scores-scores.min(1, keepdims=True))/temperature
    if correction is not None:
        logits = logits+correction
    logits = np.where(accepted, logits, -np.inf)
    valid = accepted.any(1)
    weights = np.zeros_like(scores, dtype=float)
    selected = logits[valid]
    exponent = np.exp(selected-selected.max(1, keepdims=True))
    weights[valid] = exponent/exponent.sum(1, keepdims=True)
    return weights


def mixture_diagnostics(weights, bank, failure, truth, broken, band_mass=.9):
    weights, bank = np.asarray(weights, dtype=float), np.asarray(bank, dtype=float)
    if (weights < 0).any() or not np.isclose(weights.sum(), 1) or not np.isfinite([*weights, *bank.ravel()]).all():
        raise ValueError("Invalid finite-mixture forecast")
    mean = weights@bank
    map_gap = bank[int(np.argmax(weights))]
    pair_l1 = np.abs(bank[:, None]-bank[None]).mean(-1)
    pair_l2 = np.sqrt(((bank[:, None]-bank[None])**2).mean(-1))
    crps = weights@np.abs(bank-truth).mean(-1)-.5*weights@pair_l1@weights
    energy = weights@np.sqrt(((bank-truth)**2).mean(-1))-.5*weights@pair_l2@weights
    order = np.argsort(bank, axis=0)
    ordered = np.take_along_axis(bank, order, axis=0)
    cdf = np.cumsum(weights[order], axis=0)
    lower = ordered[np.argmax(cdf >= (1-band_mass)/2, axis=0), np.arange(bank.shape[1])]
    upper = ordered[np.argmax(cdf >= 1-(1-band_mass)/2, axis=0), np.arange(bank.shape[1])]
    probability = float(weights@np.asarray(failure, dtype=float))
    positive = weights > 0
    return {"mean_gap_mse_m2": float(np.mean((mean-truth)**2)),
        "map_gap_mse_m2": float(np.mean((map_gap-truth)**2)), "path_energy_m": float(energy),
        "marginal_crps_m": float(crps), "failure_probability": probability,
        "failure_brier": float((probability-float(broken))**2),
        "candidate_band_time_coverage": float(np.mean((truth >= lower)&(truth <= upper))),
        "candidate_band_mean_width_m": float((upper-lower).mean()),
        "state_entropy_nats": float(-(weights[positive]*np.log(weights[positive])).sum()),
        "effective_candidates": float(1/(weights@weights)), "maximum_weight": float(weights.max())}
