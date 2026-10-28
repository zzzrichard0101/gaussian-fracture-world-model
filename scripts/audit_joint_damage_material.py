"""Post-run integrity checks, including independent scalar replay of a new branch."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from history_refinement import digest, write_json
from joint_damage_material import sealed
from joint_damage_material_evaluate import read_forecasts
from wmfracture.cohesive import InterfaceLaw, InterfaceState, TwoBlockSystem
from wmfracture.cohesive_inference import known_action_rollout
from wmfracture.joint_state import material_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--branch", required=True, help="Stem of the saved branch files")
    args = parser.parse_args()
    root = args.run
    packet, _, _ = read_forecasts(root)
    manifest = json.loads((root/"public-manifest.json").read_text())
    report = json.loads((root/"report.json").read_text())
    config = json.loads((root/"config.json").read_text())
    assert digest(root/"public-manifest.json") == packet["manifest_sha256"]
    assert digest(root/"inference-settings.json") == packet["settings_sha256"]
    observations = {}
    for case in manifest["cases"]:
        assert set(case) == {"id", "probe", "rgbd_file"}
        assert digest(root/case["rgbd_file"]) == packet["input_hashes"][case["id"]]
        with np.load(root/case["rgbd_file"], allow_pickle=False) as archive:
            assert set(archive.files) == {"rgb", "depth", "alpha"}
            key = hashlib.sha256(b"".join(archive[k].tobytes() for k in ("rgb", "depth", "alpha"))).hexdigest()
        observations.setdefault(key, []).append(case)
    # Current cache uses clip content as its key; verify no clip collides across actions.
    assert all(len({c["probe"] for c in cases}) == 1 for cases in observations.values())
    assert "fracture_energy" not in manifest["known_model"]["law"]
    budgets, unique = [], []
    by_id = {c["id"]: c for c in packet["cases"]}
    for case in packet["cases"]:
        for method, record in case["methods"].items():
            budgets.append(record["budget_audit"]["callback_evaluations"])
            unique.append(record["budget_audit"]["distinct_locations"])
            assert budgets[-1] <= config["budget"] and unique[-1] <= budgets[-1]
            assert len(record["parameters"]) == len(record["cutoff_states"])
    equivalent_pairs = 0
    for pair in report["paired_observation_audit"]:
        if not pair["observed_arrays_bitwise_identical"]:
            continue
        equivalent_pairs += 1
        left, right = [by_id[pair["members"][m]] for m in ("low", "high")]
        for method in config["methods"]:
            for key in ("parameters", "cutoff_states", "accepted_indices", "representative_index"):
                assert left["methods"][method][key] == right["methods"][method][key]
            assert pair["representative_prediction_max_difference_m"][method] == 0.
    branch_path = root/"branches"/args.branch
    branch = json.loads(Path(str(branch_path)+".json").read_text())
    assert branch["selection_sha256"] == digest(root/"fitted/selection.json")
    record = by_id[branch["case"]]["methods"][branch["method"]]
    with np.load(Path(str(branch_path)+".npz"), allow_pickle=False) as archive:
        parameters, gaps, failed = [archive[k] for k in ("parameters", "gaps", "failure")]
        assert np.array_equal(archive["accepted_indices"], record["accepted_indices"])
    np.testing.assert_array_equal(parameters, record["parameters"])
    checked = sorted({0, len(parameters)-1, record["representative_index"]})
    max_error = 0.
    for row in checked:
        model = material_model(packet["known_model"], parameters[row, 1])
        system = TwoBlockSystem(InterfaceLaw(**model["law"]), InterfaceState(**record["cutoff_states"][row]),
                                model["mass_left"], model["mass_right"], model["width"])
        reference = known_action_rollout(system, packet["observation_end"], max(config["target_times"]),
            config["inference_dt"], [{"id": "new", "time": branch["impact_time"], "left": branch["impact"], "right": 0.}],
            config["target_times"])
        error = float(np.max(np.abs(gaps[row]-[r["opening"] for r in reference])))
        max_error = max(max_error, error)
        assert error < 2e-10
        assert bool(failed[row]) == (reference[-1]["damage"] == 1.)
    sealed(root)
    audit = {"status": "passed", "run_directory": str(root), "public_input_hashes_unchanged": True,
        "public_material_energy_absent": True, "public_cases_exclude_pair_and_truth": True,
        "identical_clips_have_same_known_probe_in_this_run": True,
        "evaluation_counts": sorted(set(budgets)), "distinct_locations": sorted(set(unique)),
        "identical_observation_pairs_with_identical_hypotheses": equivalent_pairs,
        "selection_unchanged_after_new_branch": True, "branch": branch,
        "independent_scalar_branch_candidates_checked": checked, "maximum_scalar_branch_gap_difference_m": max_error}
    write_json(root/"integrity-checks.json", audit)
    write_json(Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/joint-damage-material/integrity-checks.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
