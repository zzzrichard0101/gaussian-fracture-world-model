"""Deploy a sealed compressed state set on a new user-specified impulse."""
import argparse
import json
from pathlib import Path

import numpy as np

from history_refinement import digest, write_json
from response_diversity import POOL_FILES, SELECTION_FILES, protect_inputs, read_sealed
from wmfracture.cohesive import InterfaceState
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.joint_state import material_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case", default="case-000")
    parser.add_argument("--pool", default="uniform49")
    parser.add_argument("--rule", default="response8")
    parser.add_argument("--impact", type=float, required=True)
    parser.add_argument("--impact-time", type=float, default=.044)
    args = parser.parse_args()
    root = args.run
    reads = protect_inputs(root, "forecast", POOL_FILES | SELECTION_FILES | {"design-plans.json"})
    pool = read_sealed(root/"fitted/pool.json")
    selected = read_sealed(root/"compressed/selection.json")
    assert selected["pool_sha256"] == digest(root/"fitted/pool.json")
    plans = json.loads((root/"design-plans.json").read_text())
    if not np.isfinite([args.impact, args.impact_time]).all() or not pool["observation_end"] < args.impact_time < max(plans["target_times"]):
        parser.error("Impulse must be finite and after observation, before the forecast horizon")
    case = next(c for c in pool["cases"] if c["id"] == args.case)
    source = case["pools"][args.pool]
    chosen = next(c for c in selected["cases"] if c["id"] == args.case)["pools"][args.pool][args.rule]
    if not chosen:
        raise ValueError("No observation-compatible physical hypotheses")
    states = [InterfaceState(**source["cutoff_states"][i]) for i in chosen]
    parameters = np.array(source["parameters"])[chosen]
    values = batch_rollout(material_model(pool["known_model"], 2.), states, np.zeros(len(states)), np.full(len(states), args.impact),
        plans["dt"], plans["target_times"], future_time=args.impact_time, start_elapsed=pool["observation_end"], fracture_energies=parameters[:, 1])
    out = root/"user-branches"; out.mkdir(exist_ok=True)
    stem = f"{args.case}-{args.pool}-{args.rule}-dv-{args.impact:g}-t-{args.impact_time:g}"
    np.savez_compressed(out/(stem+".npz"), **values, parameters=parameters, source_indices=np.array(chosen), target_times=plans["target_times"])
    write_json(out/(stem+".json"), {"pool_sha256": digest(root/"fitted/pool.json"), "compression_sha256": digest(root/"compressed/selection.json"),
        "case": args.case, "pool": args.pool, "rule": args.rule, "impact": args.impact, "impact_time": args.impact_time,
        "physical_trajectories_computed": len(chosen), "source_indices": chosen, "material_and_cutoff_state_preserved": True,
        "unique_material_identification_claimed": False, "private_reference_available": False, "experiment_files_read": sorted(reads)})
    print(f"COMPRESSED_BRANCH={out/stem}", flush=True)


if __name__ == "__main__":
    main()
