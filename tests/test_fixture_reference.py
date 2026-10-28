import json
from pathlib import Path

import pytest

from wmfracture.fixture_reference import reference_parameters
from wmfracture.joint_state import matched_history


def test_historical_fixture_overlap_check_needs_only_versioned_configs(tmp_path):
    root = Path(__file__).parents[1]
    known = json.loads((root/"config/cohesive-probe.json").read_text())
    del known["law"]["fracture_energy"]
    config = {"effective_stiffness_ratios": [.08, .12], "low_energy_base": 1.37, "low_energy_step": .021,
              "high_energy_base": 2.63, "high_energy_step": .019}
    path = tmp_path/"fixture.json"; path.write_text(json.dumps(config))
    rows = reference_parameters(path, known)
    assert [r["energy"] for r in rows] == pytest.approx([1.37, 2.63, 1.391, 2.649])
    assert rows[0]["peak"] == matched_history(known, 1.37, .08)
    assert len(list(tmp_path.iterdir())) == 1
