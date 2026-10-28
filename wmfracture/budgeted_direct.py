"""Hard-capped SciPy DIRECT callbacks; return only actually observed samples."""
import numpy as np


class BudgetExhausted(Exception):
    pass


def run_direct(objective, bounds, budget, locally_biased, options):
    from scipy.optimize import direct
    samples, values = [], []
    budget_flag = False
    def callback(x):
        nonlocal budget_flag
        if len(samples) >= budget:
            budget_flag = True
            raise BudgetExhausted("observation evaluation budget reached")
        value = float(objective(np.asarray(x, dtype=float).copy()))
        if not np.isfinite(value):
            raise ValueError("Nonfinite DIRECT observation objective")
        samples.append(np.asarray(x, dtype=float).tolist()); values.append(value)
        return value
    status = None
    try:
        result = direct(callback, bounds, maxfun=budget, locally_biased=locally_biased, **options)
        status = {"kind": "scipy_stop", "success": bool(result.success), "message": str(result.message), "scipy_nfev": int(result.nfev)}
    except (BudgetExhausted, SystemError):
        if not budget_flag:
            raise
        status = {"kind": "hard_callback_budget", "success": False, "message": "Truncated at declared callback budget; not a convergence claim", "scipy_nfev": None}
    if len(samples) > budget or not samples:
        raise ValueError("Invalid DIRECT budget accounting")
    return {"samples": samples, "scores": values, "callback_evaluations": len(samples),
        "distinct_locations": len({tuple(x) for x in samples}), "termination": status,
        "best_sample_index": int(np.argmin(values))}
