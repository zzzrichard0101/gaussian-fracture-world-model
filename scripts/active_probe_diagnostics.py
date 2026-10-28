"""Post-hoc failure diagnostics; no policy, threshold or state selection changes."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from history_refinement import write_json
from response_diversity import read_sealed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    root = parser.parse_args().run
    decisions = read_sealed(root/"planning/decision.json")["records"]
    guarded = [r for r in decisions if r["policy"] == "guarded"]
    full = {(r["case"], r["pool"]): r for r in guarded if r["state_set"] == "full"}
    planning = []
    for r in guarded:
        safe = np.array(r["eligible_actions"])
        gains = np.array(r["task_gain_over_passive"])
        action = r["selected_action_index"]
        planning.append({"case": r["case"], "pool": r["pool"], "state_set": r["state_set"],
            "selected_action": r["selected_impulse"], "positive_gain_feasible_actions": int(np.sum(safe & (gains > 1e-8))),
            "selected_max_damage_with_planning_set": r["max_candidate_added_damage"][action],
            "selected_max_damage_with_full_set": full[(r["case"], r["pool"])]["max_candidate_added_damage"][action]})
    index = json.loads((root/"active-forecast/index.json").read_text())
    targets = json.loads((root/"private/action-target-index.json").read_text())
    lookup = {(r["case"], r["action"], r["impact"]): i for i, r in enumerate(targets)}
    with np.load(root/"active-forecast/bank.npz", allow_pickle=False) as archive:
        predicted = archive["gaps"]
    with np.load(root/"private/action-targets.npz", allow_pickle=False) as archive:
        actual = archive["gaps"]
    filtered = []
    for r in index:
        candidates = np.array(r["candidate_rows"])
        target = actual[lookup[(r["case"], r["probe_action"], r["impact"])]]
        errors = np.mean((predicted[candidates]-target)**2, axis=1)
        posterior = r["posterior_local_indices"]
        filtered.append({"case": r["case"], "pool": r["pool"], "state_set": r["state_set"], "policy": r["policy"],
            "impact": r["impact"], "before_nearest_mse": float(errors.min()),
            "after_nearest_mse": float(errors[posterior].min()) if posterior else None,
            "evaluation_nearest_candidate_removed": int(np.argmin(errors)) not in posterior})
    summaries = []
    for pool in ("uniform49", "direct49"):
        for state_set in ("full", "response8"):
            p = [r for r in planning if (r["pool"], r["state_set"]) == (pool, state_set)]
            for policy in ("none", "weak", "strong", "guarded"):
                f = [r for r in filtered if (r["pool"], r["state_set"], r["policy"]) == (pool, state_set, policy)]
                good = [r for r in f if r["after_nearest_mse"] is not None]
                summaries.append({"pool": pool, "state_set": state_set, "policy": policy,
                    "before_nearest_candidate_rmse_mm": float(np.sqrt(np.mean([r["before_nearest_mse"] for r in good]))*1000),
                    "after_nearest_candidate_rmse_mm": float(np.sqrt(np.mean([r["after_nearest_mse"] for r in good]))*1000),
                    "evaluation_nearest_candidate_removed_fraction": float(np.mean([r["evaluation_nearest_candidate_removed"] for r in good])),
                    "guarded_passive_choices": sum(r["selected_action"] == 0. for r in p),
                    "guarded_episodes_without_positive_feasible_gain": sum(r["positive_gain_feasible_actions"] == 0 for r in p)})
    report = {"scope": "Post-hoc diagnostics only; no trial re-selection, parameter tuning or new validation claim.",
        "guarded_decisions": planning, "filter_summaries": summaries,
        "limitations": ["Full-set predicted damage is itself not a true-state safety guarantee.",
            "Nearest-to-truth candidate removal is a privileged evaluation diagnosis and cannot guide deployed filtering."]}
    write_json(root/"failure-diagnostics.json", report)
    write_json(Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/active-probe/failure-diagnostics.json", report)
    print(json.dumps({"filter_summaries": summaries, "planning_example": planning[:2]}), flush=True)


if __name__ == "__main__":
    main()
