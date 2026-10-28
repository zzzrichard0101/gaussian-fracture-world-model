"""Run a new impulse from the physically evolved, observation-updated state set."""
import argparse
import json
from pathlib import Path

import numpy as np

from active_probe import UPDATE_FILES
from history_refinement import digest, write_json
from response_diversity import protect_inputs, read_sealed
from wmfracture.cohesive import InterfaceState
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.joint_state import material_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case", default="case-000")
    parser.add_argument("--pool", default="uniform49")
    parser.add_argument("--state-set", default="response8")
    parser.add_argument("--policy", default="guarded")
    parser.add_argument("--impact", type=float, required=True)
    parser.add_argument("--impact-time", type=float, default=.064)
    args = parser.parse_args()
    root = args.run
    reads = protect_inputs(root, "forecast", UPDATE_FILES | {"design-plans.json"})
    updated = read_sealed(root/"updated/state.json")
    plans = json.loads((root/"design-plans.json").read_text())
    if not np.isfinite([args.impact, args.impact_time]).all() or not updated["observation_end"] < args.impact_time < max(plans["target_times"]):
        parser.error("Impulse must be finite and between the updated cutoff and horizon")
    record = next(r for r in updated["records"] if (r["case"], r["pool"], r["state_set"], r["policy"]) == (args.case, args.pool, args.state_set, args.policy))
    chosen = record["posterior_local_indices"]
    if not chosen:
        raise ValueError("No hypothesis matches the new observations; no forced forecast")
    states = [InterfaceState(**record["cutoff_states"][i]) for i in chosen]
    parameters = np.array(record["parameters"])[chosen]
    result = batch_rollout(material_model(updated["known_model"], 2.), states, np.zeros(len(states)), np.full(len(states), args.impact),
        plans["dt"], plans["target_times"], future_time=args.impact_time, start_elapsed=updated["observation_end"], fracture_energies=parameters[:, 1])
    out = root/"user-branches"; out.mkdir(exist_ok=True)
    stem = f"{args.case}-{args.pool}-{args.state_set}-{args.policy}-dv-{args.impact:g}-t-{args.impact_time:g}"
    np.savez_compressed(out/(stem+".npz"), **result, parameters=parameters, local_indices=np.array(chosen), target_times=plans["target_times"])
    write_json(out/(stem+".json"), {"updated_sha256": digest(root/"updated/state.json"), "case": args.case, "pool": args.pool,
        "state_set": args.state_set, "policy": args.policy, "observed_probe": record["selected_impulse"],
        "impact": args.impact, "impact_time": args.impact_time, "physical_trajectories": len(states), "local_indices": chosen,
        "material_and_probe_evolved_state_preserved": True, "private_reference_available": False, "experiment_files_read": sorted(reads)})
    print(f"ACTIVE_BRANCH={out/stem}", flush=True)


if __name__ == "__main__":
    main()
