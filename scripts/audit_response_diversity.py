"""Verify compression deployment against independent scalar material laws."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from history_refinement import digest, write_json
from response_diversity import read_sealed
from wmfracture.cohesive import InterfaceLaw, InterfaceState, TwoBlockSystem
from wmfracture.cohesive_inference import known_action_rollout
from wmfracture.joint_state import material_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    root = args.run
    pool = read_sealed(root/"fitted/pool.json")
    selection = read_sealed(root/"compressed/selection.json")
    branch_path = root/"user-branches"/args.branch
    branch = json.loads(Path(str(branch_path)+".json").read_text())
    assert branch["pool_sha256"] == digest(root/"fitted/pool.json") == selection["pool_sha256"]
    assert branch["compression_sha256"] == digest(root/"compressed/selection.json")
    assert not any(name.startswith("private/") or name in {"config.json", "future-plans.json"} for name in branch["experiment_files_read"])
    for relative, expected in json.loads((root/"code-sha256.json").read_text()).items():
        assert digest(root/"source-snapshot"/relative) == expected
        assert digest(Path(os.environ["WM_PROJECT_ROOT"])/relative) == expected
    case = next(c for c in pool["cases"] if c["id"] == branch["case"])
    source = case["pools"][branch["pool"]]
    chosen = next(c for c in selection["cases"] if c["id"] == branch["case"])["pools"][branch["pool"]][branch["rule"]]
    with np.load(Path(str(branch_path)+".npz"), allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    assert chosen == branch["source_indices"] == values["source_indices"].tolist()
    np.testing.assert_array_equal(values["parameters"], np.array(source["parameters"])[chosen])
    plans = json.loads((root/"design-plans.json").read_text())
    maximum = 0.
    for row, index in enumerate(chosen):
        model = material_model(pool["known_model"], source["parameters"][index][1])
        system = TwoBlockSystem(InterfaceLaw(**model["law"]), InterfaceState(**source["cutoff_states"][index]),
            model["mass_left"], model["mass_right"], model["width"])
        reference = known_action_rollout(system, pool["observation_end"], max(plans["target_times"]), plans["dt"],
            [{"id": "new", "time": branch["impact_time"], "left": branch["impact"], "right": 0.}], plans["target_times"])
        for key in ("opening", "x_left", "x_right", "v_left", "v_right", "maximum_opening", "external_work", "damage", "time", "step_index"):
            error = float(np.max(np.abs(values[key][row]-[r[key] for r in reference])))
            maximum = max(maximum, error)
            assert error < 2e-10
    audit = {"status": "passed", "run_directory": str(root), "source_snapshots_and_current_source_hashes_match": True,
        "pool_and_compression_seals_unchanged_after_deployment": True, "deployment_did_not_read_private_labels_or_test_plan": True,
        "physical_deployment_trajectories": len(chosen), "all_selected_materials_and_cutoff_states_checked": True,
        "maximum_scalar_difference_across_state_fields": maximum, "new_impulse": {"magnitude": branch["impact"], "time": branch["impact_time"]}}
    write_json(root/"integrity-checks.json", audit)
    out = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/response-diversity"
    write_json(out/"integrity-checks.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
