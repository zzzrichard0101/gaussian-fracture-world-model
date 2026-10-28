"""Closed-loop finite-set probing; trusted environment and inference separated."""
import argparse
from collections import OrderedDict
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from history_refinement import digest, write_json, launch
from joint_damage_material import generate as generate_joint
from response_diversity import (infer, design, compress, read_sealed, seal_json, read_bank,
    protect_inputs, POOL_FILES, DESIGN_FILES, SELECTION_FILES)
from wmfracture.active_probe import rgbd_summary, task_weights, select_probe
from wmfracture.cohesive import InterfaceState
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.cohesive_inference import Observation, accepted_indices, frame_residual, reachable_state
from wmfracture.joint_state import material_model


PLAN_FILES = {"planning/bank.npz", "planning/index.json", "planning/decision.json", "planning/decision.json.sha256"}
UPDATE_FILES = {"updated/state.json", "updated/state.json.sha256"}


def generate(root):
    from cohesive_probe import BlockObservations
    generate_joint(root)
    config = json.loads((root/"config.json").read_text())
    write_json(root/"compression-settings.json", {"schema": "physical-compression-1", **config["compression"], "bounds": config["parameter_bounds"]})
    write_json(root/"design-plans.json", {"impacts": config["design_impacts"], "time": config["future_time"],
        "target_times": config["target_times"], "dt": config["inference_dt"]})
    manifest = json.loads((root/"public-manifest.json").read_text())
    active = {**config["active"], "dt": config["inference_dt"], "noise": config["noise"], "acceptance": config["acceptance"]}
    write_json(root/"probe-settings.json", active)
    labels = json.loads((root/"private/parameters.json").read_text())
    states, energies, cases, actions = [], [], [], []
    # Trusted producer: physical histories/labels are never read by the planner/updater.
    for case in manifest["cases"]:
        label = labels[case["id"]]
        release = reachable_state(material_model(manifest["known_model"], label["energy"]), label["peak"]).state
        prefix = batch_rollout(material_model(manifest["known_model"], 2.), [release], [case["probe"]], [0.],
            config["truth_dt"], [max(manifest["times"])], probe_time=manifest["probe_time"], fracture_energies=[label["energy"]])
        state = {k: (int(prefix[k][0, -1]) if k == "step_index" else float(prefix[k][0, -1])) for k in asdict(release)}
        for action in active["actions"]:
            states.append(InterfaceState(**state)); energies.append(label["energy"]); cases.append(case["id"]); actions.append(action)
    probe = batch_rollout(material_model(manifest["known_model"], 2.), states, np.zeros(len(states)), actions,
        config["truth_dt"], active["observation_times"], future_time=active["action_time"],
        start_elapsed=max(manifest["times"]), fracture_energies=energies)
    observer = BlockObservations(manifest["known_model"])
    out = root/"private/extra-observations"; out.mkdir()
    final_states, future_energies, impacts, target_index, costs = [], [], [], [], []
    for row, (case, action, energy) in enumerate(zip(cases, actions, energies)):
        label = labels[case]
        # Common random numbers across probe choices and matched pair members.
        rng = np.random.default_rng(config["seed"]+3000+label["pair"])
        frames = [observer.capture({"x_left": probe["x_left"][row, t], "x_right": probe["x_right"][row, t]}) for t in range(len(active["observation_times"]))]
        arrays = {}
        for key, name in (("rgb", "rgb_sigma"), ("depth", "depth_sigma_m"), ("object_alpha", "object_alpha_sigma")):
            values = np.stack([frame[key] for frame in frames])
            values = values+rng.normal(0., config["noise"][name], values.shape)
            arrays["alpha" if key == "object_alpha" else key] = (values if key == "depth" else np.clip(values, 0, 1)).astype(np.float32)
        action_index = active["actions"].index(action)
        np.savez_compressed(out/f"{case}-a{action_index}.npz", **arrays)
        costs.append({"case": case, "action": action, "damage": float(probe["damage"][row, -1]),
            "failure": bool(probe["damage"][row, -1] == 1.)})
        state = {k: (int(probe[k][row, -1]) if k == "step_index" else float(probe[k][row, -1])) for k in asdict(states[row])}
        for impact in config["impacts"]:
            final_states.append(InterfaceState(**state)); future_energies.append(energy); impacts.append(impact)
            target_index.append({"case": case, "action": action, "impact": impact})
    future = batch_rollout(material_model(manifest["known_model"], 2.), final_states, np.zeros(len(final_states)), impacts,
        config["truth_dt"], config["target_times"], future_time=config["future_time"],
        start_elapsed=max(active["observation_times"]), fracture_energies=future_energies)
    np.savez_compressed(root/"private/action-targets.npz", gaps=future["opening"], failure=future["damage"][:, -1] == 1.,
        x_left=future["x_left"], x_right=future["x_right"])
    write_json(root/"private/action-target-index.json", target_index)
    write_json(root/"private/probe-costs.json", costs)
    print(f"ACTIVE_ENV_READY initial_cases={len(manifest['cases'])} action_specific_targets={len(target_index)}", flush=True)


def plan(root):
    allowed = POOL_FILES | DESIGN_FILES | SELECTION_FILES | PLAN_FILES | {"probe-settings.json"}
    reads = protect_inputs(root, "compress", allowed)
    began = time.perf_counter()
    pool = read_sealed(root/"fitted/pool.json")
    selection = read_sealed(root/"compressed/selection.json")
    settings = json.loads((root/"probe-settings.json").read_text())
    design_index, design_values = read_bank(root, "design")
    design_lookup = {(r["case"], r["pool"]): np.array(r["rows"]) for r in design_index}
    states, energies, actions, lookup, index = [], [], [], {}, []
    for case in pool["cases"]:
        for name, source in case["pools"].items():
            matrix = []
            for parameters, state in zip(source["parameters"], source["cutoff_states"]):
                row = []
                for action in settings["actions"]:
                    key = (*parameters, case["probe"], action)
                    if key not in lookup:
                        lookup[key] = len(states)
                        states.append(InterfaceState(**state)); energies.append(parameters[1]); actions.append(action)
                    row.append(lookup[key])
                matrix.append(row)
            index.append({"case": case["id"], "pool": name, "rows": matrix})
    bank = batch_rollout(material_model(pool["known_model"], 2.), states, np.zeros(len(states)), actions,
        settings["dt"], settings["observation_times"], future_time=settings["action_time"],
        start_elapsed=pool["observation_end"], fracture_energies=energies)
    from cohesive_probe import BlockObservations
    observer = BlockObservations(pool["known_model"])
    features = []
    for row in range(len(states)):
        features.append([rgbd_summary(observer.capture({"x_left": bank["x_left"][row, t], "x_right": bank["x_right"][row, t]}))
            for t in range(len(settings["observation_times"]))])
        if row % 200 == 0:
            print(f"ACTIVE_PLAN_RENDER {row}/{len(states)}", flush=True)
    bank["features"] = np.array(features)
    out = root/"planning"; out.mkdir()
    np.savez_compressed(out/"bank.npz", **bank)
    write_json(out/"index.json", index)
    bank_lookup = {(r["case"], r["pool"]): np.array(r["rows"]) for r in index}
    compressed = {c["id"]: c for c in selection["cases"]}
    decisions = []
    for case in pool["cases"]:
        for name, source in case["pools"].items():
            original = design_lookup[(case["id"], name)]
            nonce = int(hashlib.sha256(json.dumps(source["metrics"], sort_keys=True).encode()).hexdigest()[:8], 16)
            for state_set in settings["state_sets"]:
                chosen = compressed[case["id"]]["pools"][name][state_set]
                if not chosen:
                    raise ValueError("No matched hypotheses for active probing")
                rows = bank_lookup[(case["id"], name)][chosen]
                weights = task_weights(design_values["gaps"][original[chosen]], design_values["failure"][original[chosen]], settings["task_gap_scale_m"])
                for policy in settings["policies"]:
                    rule, seed = ("random", int(policy.split("-")[-1])+nonce) if policy.startswith("random-") else (policy, 0)
                    decision = select_probe(settings["actions"], bank["features"][rows], bank["damage"][rows, -1], rule,
                        settings["feature_scales"], settings["damage_cap"], weights, seed)
                    decisions.append({"case": case["id"], "pool": name, "state_set": state_set, "policy": policy,
                        "prior_indices": chosen, "prior_point_index": min(chosen, key=lambda i: (source["metrics"][i]["score"], source["parameters"][i])),
                        **decision})
    seal_json(out/"decision.json", {"records": decisions, "known_model": pool["known_model"], "observation_end": max(settings["observation_times"]),
        "pool_sha256": digest(root/"fitted/pool.json"), "compression_sha256": digest(root/"compressed/selection.json"),
        "settings_sha256": digest(root/"probe-settings.json"), "bank_sha256": digest(out/"bank.npz"),
        "index_sha256": digest(out/"index.json"), "seconds": time.perf_counter()-began,
        "unique_candidate_action_trajectories": len(states), "gaussian_planning_frames": len(states)*len(settings["observation_times"])})
    write_json(root/"plan-read-audit.json", sorted(reads))
    print("ACTIVE_POLICY_SEALED", flush=True)


def observe(root):
    # This stage is the trusted simulated environment, not an inference component.
    decision = read_sealed(root/"planning/decision.json")
    out = root/"public-extra"; out.mkdir()
    records, seen = [], set()
    for row in decision["records"]:
        key = row["case"], row["selected_action_index"]
        if key in seen:
            continue
        seen.add(key)
        name = f"{key[0]}-a{key[1]}.npz"
        shutil.copy2(root/"private/extra-observations"/name, out/name)
        records.append({"case": key[0], "action_index": key[1], "path": "public-extra/"+name, "sha256": digest(out/name)})
    seal_json(root/"new-observations.json", {"decision_sha256": digest(root/"planning/decision.json"), "observations": records})
    print(f"ACTIVE_OBSERVATIONS_REVEALED clips={len(records)}", flush=True)


def update(root):
    allowed = POOL_FILES | PLAN_FILES | UPDATE_FILES | {"probe-settings.json", "new-observations.json", "new-observations.json.sha256"}
    # Enumerate published names from a sealed manifest before installing its path whitelist.
    observed = read_sealed(root/"new-observations.json")
    allowed |= {row["path"] for row in observed["observations"]}
    reads = protect_inputs(root, "compress", allowed)
    observed = read_sealed(root/"new-observations.json")
    decision = read_sealed(root/"planning/decision.json")
    assert observed["decision_sha256"] == digest(root/"planning/decision.json")
    assert decision["bank_sha256"] == digest(root/"planning/bank.npz")
    assert decision["index_sha256"] == digest(root/"planning/index.json")
    settings = json.loads((root/"probe-settings.json").read_text())
    pool = read_sealed(root/"fitted/pool.json")
    cases = {c["id"]: c for c in pool["cases"]}
    index = json.loads((root/"planning/index.json").read_text())
    lookup = {(r["case"], r["pool"]): np.array(r["rows"]) for r in index}
    with np.load(root/"planning/bank.npz", allow_pickle=False) as archive:
        bank = {k: archive[k] for k in archive.files}
    observation_lookup, clips = {}, {}
    for row in observed["observations"]:
        assert digest(root/row["path"]) == row["sha256"]
        with np.load(root/row["path"], allow_pickle=False) as archive:
            assert set(archive.files) == {"rgb", "depth", "alpha"}
            arrays = {k: archive[k] for k in archive.files}
        content_hash = hashlib.sha256(b"".join(arrays[k].tobytes() for k in ("rgb", "depth", "alpha"))).hexdigest()
        observation_lookup[(row["case"], row["action_index"])] = content_hash
        clips[content_hash] = [Observation(t, rgb, depth, alpha) for t, rgb, depth, alpha in zip(settings["observation_times"], arrays["rgb"], arrays["depth"], arrays["alpha"])]
    from cohesive_probe import BlockObservations
    observer = BlockObservations(pool["known_model"])
    cache, frames, records = {}, OrderedDict(), []
    fields = tuple(asdict(InterfaceState(0., 0.)))
    for entry in decision["records"]:
        source = cases[entry["case"]]["pools"][entry["pool"]]
        content_hash = observation_lookup[(entry["case"], entry["selected_action_index"])]
        metrics, states = [], []
        for index in entry["prior_indices"]:
            row = int(lookup[(entry["case"], entry["pool"])][index, entry["selected_action_index"]])
            key = content_hash, row
            if key not in cache:
                if row not in frames:
                    frames[row] = [observer.capture({"x_left": bank["x_left"][row, t], "x_right": bank["x_right"][row, t]}) for t in range(len(settings["observation_times"]))]
                    if len(frames) > 64:
                        frames.popitem(last=False)
                residuals = [frame_residual(a, b, settings["noise"]) for a, b in zip(clips[content_hash], frames[row])]
                fresh = {k: float(np.mean([r[k] for r in residuals])) for k in residuals[0]}
                cache[key] = fresh
            previous, fresh = source["metrics"][index], cache[key]
            metrics.append({k: ((fresh[k]+previous[k])/2 if k == "score" else np.sqrt((fresh[k]**2+previous[k]**2)/2)) for k in fresh})
            states.append({k: (int(bank[k][row, -1]) if k == "step_index" else float(bank[k][row, -1])) for k in fields})
        accepted = accepted_indices(metrics, settings["acceptance"])
        after_point = min(accepted, key=lambda i: (metrics[i]["score"], source["parameters"][entry["prior_indices"][i]])) if accepted else None
        records.append({**entry, "posterior_local_indices": accepted, "posterior_point_local_index": after_point,
            "metrics": metrics, "parameters": [source["parameters"][i] for i in entry["prior_indices"]], "cutoff_states": states})
    out = root/"updated"; out.mkdir()
    seal_json(out/"state.json", {"records": records, "known_model": pool["known_model"], "observation_end": decision["observation_end"],
        "pool_sha256": digest(root/"fitted/pool.json"), "decision_sha256": digest(root/"planning/decision.json"),
        "observations_sha256": digest(root/"new-observations.json"), "unique_clip_candidate_residuals": len(cache)})
    write_json(root/"update-read-audit.json", sorted(reads))
    print("ACTIVE_UPDATED_SEALED", flush=True)


def forecast(root):
    files = {"future-plans.json", "active-forecast/bank.npz", "active-forecast/index.json", "active-forecast/seal.json", "active-forecast/seal.json.sha256"}
    reads = protect_inputs(root, "forecast", UPDATE_FILES | files)
    updated = read_sealed(root/"updated/state.json")
    plans = json.loads((root/"future-plans.json").read_text())
    states, energies, impacts, lookup, index = [], [], [], {}, []
    for entry in updated["records"]:
        for impact in plans["impacts"]:
            rows = []
            for parameters, state in zip(entry["parameters"], entry["cutoff_states"]):
                key = (*parameters, entry["selected_impulse"], impact)
                if key not in lookup:
                    lookup[key] = len(states); states.append(InterfaceState(**state)); energies.append(parameters[1]); impacts.append(impact)
                rows.append(lookup[key])
            index.append({"case": entry["case"], "pool": entry["pool"], "state_set": entry["state_set"], "policy": entry["policy"],
                "probe_action": entry["selected_impulse"], "impact": impact, "candidate_rows": rows,
                "before_point_local_index": entry["prior_indices"].index(entry["prior_point_index"]),
                "posterior_local_indices": entry["posterior_local_indices"], "after_point_local_index": entry["posterior_point_local_index"]})
    values = batch_rollout(material_model(updated["known_model"], 2.), states, np.zeros(len(states)), impacts, plans["dt"], plans["target_times"],
        future_time=plans["time"], start_elapsed=updated["observation_end"], fracture_energies=energies)
    out = root/"active-forecast"; out.mkdir()
    np.savez_compressed(out/"bank.npz", gaps=values["opening"], failure=values["damage"][:, -1] == 1., x_left=values["x_left"], x_right=values["x_right"])
    write_json(out/"index.json", index)
    seal_json(out/"seal.json", {"updated_sha256": digest(root/"updated/state.json"), "plans_sha256": digest(root/"future-plans.json"),
        "bank_sha256": digest(out/"bank.npz"), "index_sha256": digest(out/"index.json"), "unique_physical_branches": len(states)})
    write_json(root/"active-forecast-read-audit.json", sorted(reads))
    print(f"ACTIVE_FORECAST branches={len(states)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    stages = ("generate", "infer", "design", "compress", "plan", "observe", "update", "forecast", "evaluate", "render")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--stage", choices=stages)
    args = parser.parse_args()
    if args.stage:
        if args.run is None:
            parser.error("stage requires --run")
        if args.stage in ("evaluate", "render"):
            from active_probe_evaluate import evaluate, render
            {"evaluate": evaluate, "render": render}[args.stage](args.run)
        else:
            globals()[args.stage](args.run)
        return
    project = Path(os.environ["WM_PROJECT_ROOT"])
    config = json.loads((project/"config/active-probe.json").read_text())
    root = Path(os.environ["WM_RESEARCH_ROOT"])/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-active-probe")
    root.mkdir(parents=True); write_json(root/"config.json", config)
    sources = [Path(__file__), project/"scripts/active_probe_evaluate.py", project/"scripts/response_diversity.py",
        project/"scripts/joint_damage_material.py", project/"wmfracture/active_probe.py", project/"wmfracture/cohesive_batch.py",
        project/"wmfracture/cohesive.py", project/"wmfracture/cohesive_inference.py", project/"wmfracture/hypothesis_selection.py",
        project/"wmfracture/fixture_reference.py", project/"config/joint-damage-material.json", project/"config/response-diversity.json"]
    write_json(root/"code-sha256.json", {str(p.relative_to(project)): digest(p) for p in sources})
    for path in sources:
        destination = root/"source-snapshot"/path.relative_to(project); destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    print(f"OUTPUT={root}", flush=True)
    for stage in stages:
        launch([sys.executable, "-u", str(Path(__file__)), "--stage", stage, "--run", str(root)], root/f"{stage}.log")


if __name__ == "__main__":
    main()
