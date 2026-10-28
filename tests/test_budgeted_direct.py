import numpy as np
import pytest

from wmfracture.budgeted_direct import run_direct


def test_direct_cannot_exceed_observation_budget_even_when_scipy_maxfun_is_soft():
    for local in (False, True):
        calls = []
        def objective(x):
            calls.append(x.copy())
            return float(np.sum((x-.371)**2))
        result = run_direct(objective, [(0., 1.), (0., 1.)], 49, local,
            {"eps": 1e-4, "maxiter": 1000, "vol_tol": 1e-30, "len_tol": 1e-15})
        assert len(calls) == result["callback_evaluations"] <= 49
        assert result["samples"][result["best_sample_index"]] in [x.tolist() for x in calls]
        assert min(result["scores"]) < .03


def test_unrelated_objective_failures_are_not_mislabeled_as_budget_exhaustion():
    def broken(x):
        raise ValueError("invalid observation")
    with pytest.raises((ValueError, SystemError)):
        run_direct(broken, [(0., 1.)], 49, False, {"maxiter": 1000})
