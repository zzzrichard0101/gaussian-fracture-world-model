"""Reconstruct reference fixture parameters from versioned configs, not local runs."""
import json
from pathlib import Path

from wmfracture.joint_state import matched_history


def reference_parameters(config_path, known_model):
    config = json.loads(Path(config_path).read_text())
    rows = []
    for pair, ratio in enumerate(config["effective_stiffness_ratios"]):
        for name in ("low", "high"):
            energy = config[name+"_energy_base"]+pair*config[name+"_energy_step"]
            rows.append({"energy": energy, "peak": matched_history(known_model, energy, ratio)})
    return rows
