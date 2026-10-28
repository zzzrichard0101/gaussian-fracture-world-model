"""Observation-only proposals for bounded one-dimensional reachable histories."""
import numpy as np


SEARCH_KEYS = {"lower", "upper", "initial_count", "budget", "batch_size", "minimum_interval_m", "flat_score_tolerance", "dt", "acceptance"}


def validate_search(settings):
    if set(settings) != SEARCH_KEYS:
        raise ValueError("Search settings must exclude future plans and labels")
    if not (0 <= settings["lower"] < settings["upper"] and settings["initial_count"] >= 3
            and settings["budget"] >= settings["initial_count"] and settings["batch_size"] >= 1
            and settings["minimum_interval_m"] > 0 and settings["flat_score_tolerance"] >= 0):
        raise ValueError("Invalid history-search budget or bounds")


def canonical_peak(peak):
    return round(float(peak), 15)


def initial_points(method, settings):
    validate_search(settings)
    count = settings["budget"] if method == "uniform49" else settings["initial_count"]
    return [canonical_peak(value) for value in np.linspace(settings["lower"], settings["upper"], count)]


def propose(peaks, scores, method, settings):
    """No callback or state/future input: only the evaluated prefix loss curve."""
    validate_search(settings)
    peaks, scores = np.asarray(peaks, dtype=float), np.asarray(scores, dtype=float)
    if (peaks.shape != scores.shape or len(peaks) < 3 or not np.isfinite([peaks, scores]).all()
        or len(set(peaks)) != len(peaks)):
        raise ValueError("Invalid evaluated observation curve")
    order = np.argsort(peaks)
    peaks, scores = peaks[order], scores[order]
    if peaks[0] != settings["lower"] or peaks[-1] != settings["upper"]:
        raise ValueError("Search must retain both domain endpoints")
    remaining = settings["budget"]-len(peaks)
    count = min(settings["batch_size"], remaining)
    if count <= 0:
        return []
    selected, temporary = [], peaks.tolist()
    minimum = settings["minimum_interval_m"]
    def add_midpoint(lower, upper):
        candidates = [(b-a, a, b) for a, b in zip(temporary[:-1], temporary[1:])
            if a >= lower-1e-16 and b <= upper+1e-16 and b-a > minimum]
        if not candidates:
            return False
        _, a, b = min(candidates, key=lambda item: (-item[0], item[1]))
        point = canonical_peak((a+b)/2)
        if point in temporary:
            return False
        selected.append(point); temporary.append(point); temporary.sort()
        return True
    flat = np.ptp(scores) <= settings["flat_score_tolerance"]
    if flat:
        while len(selected) < count and add_midpoint(peaks[0], peaks[-1]):
            pass
        return selected
    if method == "greedy49":
        best = int(np.argmin(scores))
        lower, upper = peaks[max(0, best-1)], peaks[min(len(peaks)-1, best+1)]
        while len(selected) < count and add_midpoint(lower, upper):
            pass
    elif method == "multiregion49":
        minima = [i for i in range(len(peaks)) if
            (i == 0 or scores[i] <= scores[i-1]) and (i == len(peaks)-1 or scores[i] <= scores[i+1])]
        minima.sort(key=lambda i: (scores[i], peaks[i]))
        refine_slots = max(0, count-1)
        for slot in range(refine_slots):
            best = minima[slot % len(minima)]
            add_midpoint(peaks[max(0, best-1)], peaks[min(len(peaks)-1, best+1)])
    else:
        raise ValueError("Only adaptive methods accept new-point proposals")
    while len(selected) < count and add_midpoint(peaks[0], peaks[-1]):
        pass
    if len(selected) != count:
        raise ValueError("Search exhausted numerical resolution before its declared budget")
    return selected


def selection(metrics, settings):
    from wmfracture.cohesive_inference import accepted_indices
    scores = np.array([item["score"] for item in metrics])
    accepted = accepted_indices(metrics, settings["acceptance"])
    ambiguous = bool(np.ptp(scores) <= settings["flat_score_tolerance"])
    point = min(accepted, key=lambda index: (scores[index], index)) if accepted and not ambiguous else None
    return {"accepted_indices": accepted, "point_index": point,
        "status": "no_observation_match" if not accepted else ("ambiguous" if ambiguous else "matched"),
        "best_observed_score": float(scores.min())}
