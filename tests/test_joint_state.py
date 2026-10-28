import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.cohesive_inference import reachable_state, known_action_rollout
from wmfracture.joint_state import material_model, matched_history, partition_search


def known():
    c = json.loads((Path(__file__).parents[1]/"config/cohesive-probe.json").read_text())
    del c["law"]["fracture_energy"]
    return c


def test_true_energy_cannot_accidentally_be_supplied_as_a_known_parameter():
    c = known(); c["law"]["fracture_energy"] = 2.
    with pytest.raises(ValueError, match="absent"):
        material_model(c, 1.3)


def test_matched_histories_have_same_unloading_stiffness_but_different_material_state():
    c = known()
    values = []
    for energy in (1.37, 2.63):
        peak = matched_history(c, energy, .08)
        law = InterfaceLaw(**material_model(c, energy)["law"])
        values.append((1-law.damage(peak))*law.tensile_stiffness)
    np.testing.assert_allclose(values, [.08*c["law"]["tensile_stiffness"]]*2, atol=1e-7)
    assert matched_history(c, 1.37, .08) != matched_history(c, 2.63, .08)


def test_heterogeneous_material_batch_matches_independent_physics():
    c = known(); states, references = [], []
    times = [0., .004, .012, .036, .04, .12, .18]
    for energy in (1.37, 2.63):
        model = material_model(c, energy)
        system = reachable_state(model, matched_history(c, energy, .08))
        states.append(copy.deepcopy(system.state))
        references.append(known_action_rollout(system, 0., .18, 5e-6,
            [{"id": "p", "time": .004, "left": .04, "right": 0.}, {"id": "f", "time": .036, "left": .265, "right": 0.}], times))
    result = batch_rollout(material_model(c, 2.), states, [.04]*2, [.265]*2, 5e-6, times, fracture_energies=[1.37, 2.63])
    for key in result:
        np.testing.assert_allclose(result[key], [[row[key] for row in r] for r in references], atol=2e-10, rtol=0)
    assert all(state.time == 1. for state in states)


def test_2d_partition_respects_budget_and_samples_multiple_regions():
    result = partition_search(lambda x: min(float(((x-[.2,.3])**2).sum()), .01+float(((x-[.8,.7])**2).sum())), 49)
    assert result["callback_evaluations"] == result["distinct_locations"] == 49
    assert len({tuple(x) for x in result["samples"]}) == 49
    assert any(step["global_split"] for step in result["trace"])
