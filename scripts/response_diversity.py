"""Observation-only pool -> simulated design responses -> sealed compression."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from history_refinement import digest, write_json, launch
from joint_damage_material import JointEngine, generate as generate_joint
from wmfracture.budgeted_direct import run_direct
from wmfracture.cohesive import InterfaceState
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.cohesive_inference import accepted_indices
from wmfracture.hypothesis_selection import response_features, select_hypotheses
from wmfracture.joint_state import material_model, physical_parameters


def protect_inputs(root, stage, allowed):
    """Deny unexpected reads within the experiment, including private labels."""
    seen = set()
    root = root.resolve()
    # Outputs may be re-read for checksums, but private/config/held-out inputs remain denied.
    output_folder = {"infer": "fitted", "design": "design", "compress": "compressed", "forecast": "forecast"}[stage]
    output_names = (["pool.json", "pool.json.sha256"] if stage == "infer" else
                    ["selection.json", "selection.json.sha256"] if stage == "compress" else
                    ["branches.npz", "index.json", "seal.json", "seal.json.sha256"])
    allowed = set(allowed) | {output_folder+"/"+name for name in output_names}
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode, flags = args[1:3]
        reading = (isinstance(mode, str) and mode.startswith("r")) or (mode is None and flags & os.O_ACCMODE != os.O_WRONLY)
        if not reading:
            return
        path = Path(os.fsdecode(args[0])).resolve()
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            return
        if relative not in allowed and not (stage == "infer" and relative.startswith("public/") and relative.endswith(".npz")):
            raise PermissionError(f"{stage} refuses undeclared experiment input: {relative}")
        seen.add(relative)
    sys.addaudithook(audit)
    return seen


def seal_json(path, value):
    write_json(path, value)
    path.with_name(path.name+".sha256").write_text(digest(path))


def read_sealed(path):
    if digest(path) != path.with_name(path.name+".sha256").read_text():
        raise ValueError(f"Sealed file changed: {path}")
    return json.loads(path.read_text())


POOL_FILES = {"fitted/pool.json", "fitted/pool.json.sha256"}
DESIGN_FILES = {"design/branches.npz", "design/index.json", "design/seal.json", "design/seal.json.sha256"}
SELECTION_FILES = {"compressed/selection.json", "compressed/selection.json.sha256"}


def generate(root):
    generate_joint(root)
    config = json.loads((root/"config.json").read_text())
    write_json(root/"compression-settings.json", {"schema": "physical-compression-1", **config["compression"],
        "bounds": config["parameter_bounds"]})
    write_json(root/"design-plans.json", {"impacts": config["design_impacts"], "time": config["future_time"],
        "target_times": config["target_times"], "dt": config["inference_dt"]})


def infer(root):
    reads = protect_inputs(root, "infer", {"public-manifest.json", "inference-settings.json"})
    began = time.perf_counter()
    manifest = json.loads((root/"public-manifest.json").read_text())
    settings = json.loads((root/"inference-settings.json").read_text())
    engine = JointEngine(root, manifest, settings)
    units = {"uniform49": [np.array([x, y]) for x in np.linspace(0, 1, 7) for y in np.linspace(0, 1, 7)]}
    for method in settings["methods"]:
        if method.startswith("random49-"):
            units[method] = list(np.random.default_rng(int(method.split("-")[-1])).random((49, 2)))
    engine.predict([(probe, *physical_parameters(unit, manifest["parameter_bounds"]))
        for probe in sorted({c["probe"] for c in manifest["cases"]}) for points in units.values() for unit in points])
    records = []
    for case in manifest["cases"]:
        pools = {}
        for method in settings["methods"]:
            def objective(x):
                return engine.evaluate(case["id"], x)[0]["score"]
            if method in units:
                points = units[method]
                run = {"samples": [x.tolist() for x in points], "scores": [objective(x) for x in points],
                    "callback_evaluations": len(points), "distinct_locations": len(points)}
            else:
                if method != "direct49":
                    raise ValueError("Unknown candidate pool")
                run = run_direct(objective, [(0., 1.)]*2, settings["budget"], False, settings["optimizer"])
            points = list(dict.fromkeys(tuple(row) for row in run["samples"]))
            parameters = [list(physical_parameters(unit, manifest["parameter_bounds"])) for unit in points]
            evaluated = [engine.evaluate(case["id"], unit) for unit in points]
            metrics = [r[0] for r in evaluated]
            pools[method] = {"parameters": parameters, "metrics": metrics, "cutoff_states": [r[1] for r in evaluated],
                "accepted_indices": accepted_indices(metrics, settings["acceptance"]), "budget_audit": run}
        records.append({"id": case["id"], "probe": case["probe"], "pools": pools})
        print(f"DIVERSITY_POOL {len(records)}/{len(manifest['cases'])}", flush=True)
    out = root/"fitted"; out.mkdir()
    seal_json(out/"pool.json", {"schema": "observation-only-physical-pool-1", "known_model": manifest["known_model"],
        "observation_end": max(manifest["times"]), "bounds": manifest["parameter_bounds"], "cases": records,
        "manifest_sha256": digest(root/"public-manifest.json"), "settings_sha256": digest(root/"inference-settings.json"),
        "input_hashes": engine.input_hashes, "cache_statistics": engine.stats, "seconds": time.perf_counter()-began})
    write_json(root/"infer-read-audit.json", sorted(reads))


def simulate(root, plans, folder, pool):
    states, energies, actions, keys, lookup, index = [], [], [], [], {}, []
    for case in pool["cases"]:
        for name, record in case["pools"].items():
            matrix = []
            for parameters, state in zip(record["parameters"], record["cutoff_states"]):
                rows = []
                for action in plans["impacts"]:
                    key = (case["probe"], *parameters, action)
                    if key not in lookup:
                        lookup[key] = len(states); keys.append(key)
                        states.append(InterfaceState(**state)); energies.append(parameters[1]); actions.append(action)
                    rows.append(lookup[key])
                matrix.append(rows)
            index.append({"case": case["id"], "probe": case["probe"], "pool": name, "rows": matrix})
    values = batch_rollout(material_model(pool["known_model"], 2.), states, np.zeros(len(states)), actions, plans["dt"],
        plans["target_times"], future_time=plans["time"], start_elapsed=pool["observation_end"], fracture_energies=energies)
    out = root/folder; out.mkdir()
    np.savez_compressed(out/"branches.npz", gaps=values["opening"], failure=values["damage"][:, -1] == 1.,
        x_left=values["x_left"], x_right=values["x_right"])
    write_json(out/"index.json", index)
    audit = {"pool_sha256": digest(root/"fitted/pool.json"),
        "plans_sha256": digest(root/("design-plans.json" if folder == "design" else "future-plans.json")),
        "branches_sha256": digest(out/"branches.npz"), "index_sha256": digest(out/"index.json"),
        "unique_physical_branches": len(states),
        "logical_trajectory_evaluations": sum(len(r["parameters"])*len(plans["impacts"]) for c in pool["cases"] for r in c["pools"].values())}
    if folder == "forecast":
        audit["compression_sha256"] = digest(root/"compressed/selection.json")
    seal_json(out/"seal.json", audit)
    print(f"DIVERSITY_{folder.upper()} branches={len(states)}", flush=True)


def design(root):
    reads = protect_inputs(root, "design", POOL_FILES | {"design-plans.json"})
    simulate(root, json.loads((root/"design-plans.json").read_text()), "design", read_sealed(root/"fitted/pool.json"))
    write_json(root/"design-read-audit.json", sorted(reads))


def read_bank(root, folder):
    seal = read_sealed(root/folder/"seal.json")
    for key, path in [("pool_sha256", root/"fitted/pool.json"), ("branches_sha256", root/folder/"branches.npz"),
                      ("index_sha256", root/folder/"index.json")]:
        if digest(path) != seal[key]:
            raise ValueError("Physical branch bank changed")
    with np.load(root/folder/"branches.npz", allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    return json.loads((root/folder/"index.json").read_text()), values


def compress(root):
    reads = protect_inputs(root, "compress", POOL_FILES | DESIGN_FILES | {"compression-settings.json"})
    pool = read_sealed(root/"fitted/pool.json")
    settings = json.loads((root/"compression-settings.json").read_text())
    index, values = read_bank(root, "design")
    lookup = {(r["case"], r["pool"]): np.array(r["rows"]) for r in index}
    records = []
    for case in pool["cases"]:
        pools = {}
        for name, record in case["pools"].items():
            rows = lookup[(case["id"], name)]
            features = response_features(values["gaps"][rows], values["failure"][rows], settings["gap_scale_m"])
            selected = {}
            for rule in ("all", "residual", "parameter", "response", "random"):
                for seed in (settings["random_seeds"] if rule == "random" else [0]):
                    label = f"random8-{seed}" if rule == "random" else ("full" if rule == "all" else rule+"8")
                    selected[label] = select_hypotheses(record["parameters"], [m["score"] for m in record["metrics"]],
                        record["accepted_indices"], settings["count"], rule, settings["bounds"], features, seed)
            pools[name] = selected
        records.append({"id": case["id"], "probe": case["probe"], "pools": pools})
    out = root/"compressed"; out.mkdir()
    seal_json(out/"selection.json", {"schema": "design-response-compression-1", "cases": records,
        "pool_sha256": digest(root/"fitted/pool.json"), "design_seal_sha256": digest(root/"design/seal.json"),
        "settings_sha256": digest(root/"compression-settings.json"), "observation_acceptance_changed": False})
    write_json(root/"compress-read-audit.json", sorted(reads))
    print("DIVERSITY_COMPRESSED", flush=True)


def forecast(root):
    reads = protect_inputs(root, "forecast", POOL_FILES | SELECTION_FILES | {"future-plans.json"})
    read_sealed(root/"compressed/selection.json")
    simulate(root, json.loads((root/"future-plans.json").read_text()), "forecast", read_sealed(root/"fitted/pool.json"))
    write_json(root/"forecast-read-audit.json", sorted(reads))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--stage", choices=["generate", "infer", "design", "compress", "forecast", "evaluate", "render"])
    args = parser.parse_args()
    if args.stage:
        if args.run is None:
            parser.error("A stage requires --run")
        if args.stage in ("evaluate", "render"):
            from response_diversity_evaluate import evaluate, render
            {"evaluate": evaluate, "render": render}[args.stage](args.run)
        else:
            globals()[args.stage](args.run)
        return
    project = Path(os.environ["WM_PROJECT_ROOT"])
    config = json.loads((project/"config/response-diversity.json").read_text())
    root = Path(os.environ["WM_RESEARCH_ROOT"])/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-response-diversity")
    root.mkdir(parents=True)
    write_json(root/"config.json", config)
    sources = [Path(__file__), project/"scripts/response_diversity_evaluate.py", project/"scripts/joint_damage_material.py",
        project/"wmfracture/hypothesis_selection.py", project/"wmfracture/cohesive_batch.py",
        project/"wmfracture/fixture_reference.py", project/"config/joint-damage-material.json"]
    write_json(root/"code-sha256.json", {str(p.relative_to(project)): digest(p) for p in sources})
    snapshots = root/"source-snapshot"; snapshots.mkdir()
    for path in sources:
        target = snapshots/path.relative_to(project); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(f"OUTPUT={root}", flush=True)
    for stage in ("generate", "infer", "design", "compress", "forecast", "evaluate", "render"):
        launch([sys.executable, "-u", str(Path(__file__)), "--stage", stage, "--run", str(root)], root/f"{stage}.log")


if __name__ == "__main__":
    main()
