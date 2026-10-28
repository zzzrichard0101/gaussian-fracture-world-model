"""Post-run physics and data-flow checks, never policy inputs."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from history_refinement import digest, write_json
from response_diversity import read_sealed
from wmfracture.cohesive import InterfaceLaw, InterfaceState, TwoBlockSystem
from wmfracture.cohesive_inference import known_action_rollout, reachable_state
from wmfracture.joint_state import material_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    root = args.run
    pool = read_sealed(root/"fitted/pool.json")
    decision = read_sealed(root/"planning/decision.json")
    updated = read_sealed(root/"updated/state.json")
    observed = read_sealed(root/"new-observations.json")
    config = json.loads((root/"config.json").read_text())
    assert decision["pool_sha256"] == updated["pool_sha256"] == digest(root/"fitted/pool.json")
    assert decision["compression_sha256"] == digest(root/"compressed/selection.json")
    for relative, expected in json.loads((root/"code-sha256.json").read_text()).items():
        assert digest(root/"source-snapshot"/relative) == digest(Path(os.environ["WM_PROJECT_ROOT"])/relative) == expected
    source_cases = {c["id"]: c for c in pool["cases"]}
    bank_index = json.loads((root/"planning/index.json").read_text())
    lookup = {(r["case"], r["pool"]): np.array(r["rows"]) for r in bank_index}
    with np.load(root/"planning/bank.npz", allow_pickle=False) as archive:
        bank = {k: archive[k] for k in archive.files}
    fields = tuple(asdict(InterfaceState(0., 0.)))
    for record in updated["records"]:
        source = source_cases[record["case"]]["pools"][record["pool"]]
        for local, index in enumerate(record["prior_indices"]):
            row = lookup[(record["case"], record["pool"])][index, record["selected_action_index"]]
            assert record["parameters"][local] == source["parameters"][index]
            for field in fields:
                assert record["cutoff_states"][local][field] == bank[field][row, -1]
    requested = {(r["case"], r["selected_action_index"]) for r in decision["records"]}
    assert requested == {(r["case"], r["action_index"]) for r in observed["observations"]}
    for row in observed["observations"]:
        assert digest(root/row["path"]) == row["sha256"]
    # Independent continuous ground truth for passive and two physical probing choices.
    labels = json.loads((root/"private/parameters.json").read_text())
    target_index = json.loads((root/"private/action-target-index.json").read_text())
    targets = {(r["case"], r["action"], r["impact"]): i for i, r in enumerate(target_index)}
    with np.load(root/"private/action-targets.npz", allow_pickle=False) as archive:
        truth = {k: archive[k] for k in archive.files}
    truth_error = 0.
    for action in (0., .20, .26):
        label = labels["case-000"]
        model = material_model(pool["known_model"], label["energy"])
        system = reachable_state(model, label["peak"])
        reference = known_action_rollout(system, 0., max(config["target_times"]), config["truth_dt"],
            [{"id": "initial", "time": .004, "left": .04, "right": 0.},
             {"id": "extra", "time": config["active"]["action_time"], "left": action, "right": 0.},
             {"id": "future", "time": config["future_time"], "left": config["impacts"][1], "right": 0.}], config["target_times"])
        target = targets[("case-000", action, config["impacts"][1])]
        error = float(np.max(np.abs(truth["gaps"][target]-[r["opening"] for r in reference])))
        truth_error = max(truth_error, error)
        assert error < 2e-10 and bool(truth["failure"][target]) == (reference[-1]["damage"] == 1.)
    branch_path = root/"user-branches"/args.branch
    branch = json.loads(Path(str(branch_path)+".json").read_text())
    assert branch["updated_sha256"] == digest(root/"updated/state.json")
    assert not any(p.startswith("private/") or p in {"future-plans.json", "config.json"} for p in branch["experiment_files_read"])
    source = next(r for r in updated["records"] if (r["case"], r["pool"], r["state_set"], r["policy"]) == (branch["case"], branch["pool"], branch["state_set"], branch["policy"]))
    with np.load(Path(str(branch_path)+".npz"), allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    assert data["local_indices"].tolist() == source["posterior_local_indices"] == branch["local_indices"]
    branch_error = 0.
    for row, index in enumerate(branch["local_indices"]):
        np.testing.assert_array_equal(data["parameters"][row], source["parameters"][index])
        model = material_model(updated["known_model"], source["parameters"][index][1])
        system = TwoBlockSystem(InterfaceLaw(**model["law"]), InterfaceState(**source["cutoff_states"][index]), model["mass_left"], model["mass_right"], model["width"])
        reference = known_action_rollout(system, updated["observation_end"], max(config["target_times"]), config["inference_dt"],
            [{"id": "new", "time": branch["impact_time"], "left": branch["impact"], "right": 0.}], config["target_times"])
        for key in fields+("opening", "damage"):
            error = float(np.max(np.abs(data[key][row]-[r[key] for r in reference])))
            branch_error = max(branch_error, error)
            assert error < 2e-10
    audit = {"status": "passed", "run_directory": str(root), "recorded_source_snapshots_and_current_hashes_match": True,
        "all_updated_hypotheses_keep_their_own_material_and_exact_probe_evolved_state": True,
        "published_observations_exactly_match_frozen_policy_requests": True,
        "original_pool_and_compression_seals_unchanged": True,
        "independent_action_specific_truth_trajectories_checked": 3, "maximum_continuous_scalar_truth_gap_difference_m": truth_error,
        "new_branch_hypotheses_checked": len(branch["local_indices"]), "maximum_scalar_new_branch_difference_across_state_fields": branch_error,
        "new_branch_excludes_private_truth_and_test_plans": True}
    write_json(root/"integrity-checks.json", audit)
    write_json(Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/active-probe/integrity-checks.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
