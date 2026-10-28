"""Unknown material + valid damage history: observed-equivalent physical twins."""
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
from wmfracture.budgeted_direct import run_direct
from wmfracture.cohesive import InterfaceState
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.cohesive_inference import Observation, accepted_indices, frame_residual, reachable_state
from wmfracture.joint_state import material_model, matched_history, partition_search, physical_parameters


def generate(root):
    from cohesive_probe import BlockObservations
    config = json.loads((root/"config.json").read_text())
    project = Path(os.environ["WM_PROJECT_ROOT"])
    source = json.loads((project/config["base_model_config"]).read_text())
    known = {k: copy.deepcopy(source[k]) for k in ("law", "mass_left", "mass_right", "width", "prehistory_duration", "prehistory_samples", "render")}
    del known["law"]["fracture_energy"]
    known["render"]["width"] = config["render_width"]
    observer = BlockObservations(known)
    (root/"public").mkdir(); (root/"private").mkdir()
    states, energies, probes, labels, public = [], [], [], {}, []
    for pair, ratio in enumerate(config["effective_stiffness_ratios"]):
        for member, energy in [("low", config["low_energy_base"]+pair*config["low_energy_step"]),
                                ("high", config["high_energy_base"]+pair*config["high_energy_step"])]:
            peak = matched_history(known, energy, ratio)
            for probe in config["probes"]:
                identifier = f"case-{len(states):03d}"
                states.append(reachable_state(material_model(known, energy), peak).state)
                energies.append(energy); probes.append(probe)
                labels[identifier] = {"pair": pair, "member": member, "energy": energy, "peak": peak,
                    "effective_stiffness_ratio": ratio, "probe": probe}
                public.append({"id": identifier, "probe": probe, "rgbd_file": f"public/{identifier}.npz"})
    prefix = batch_rollout(material_model(known, 2.), states, probes, np.zeros(len(states)), config["truth_dt"], config["context_times"],
        probe_time=config["probe_time"], future_time=config["future_time"], fracture_energies=energies)
    for row, case in enumerate(public):
        label = labels[case["id"]]
        rng = np.random.default_rng(config["seed"]+label["pair"]*10+config["probes"].index(case["probe"]))
        fields = [observer.capture({"x_left": prefix["x_left"][row, t], "x_right": prefix["x_right"][row, t]}) for t in range(len(config["context_times"]))]
        noisy = {}
        for key, noise in [("rgb", "rgb_sigma"), ("depth", "depth_sigma_m"), ("object_alpha", "object_alpha_sigma")]:
            array = np.stack([frame[key] for frame in fields])
            array = array+rng.normal(0, config["noise"][noise], array.shape)
            noisy[key] = (array if key == "depth" else np.clip(array, 0, 1)).astype(np.float32)
        np.savez_compressed(root/case["rgbd_file"], rgb=noisy["rgb"], depth=noisy["depth"], alpha=noisy["object_alpha"])
        label["prefix_maximum_opening"] = float(prefix["maximum_opening"][row, -1])
        label["probe_added_damage_history"] = bool(label["prefix_maximum_opening"] > label["peak"]+1e-12)
        label["prefix_failed"] = bool(prefix["damage"][row, -1] == 1.)
    future_states, future_energies, future_probes, future_impacts, target_index = [], [], [], [], []
    for row, case in enumerate(public):
        for impact in config["impacts"]:
            future_states.append(copy.deepcopy(states[row])); future_energies.append(energies[row])
            future_probes.append(case["probe"]); future_impacts.append(impact)
            target_index.append({"case": case["id"], "impact": impact})
    targets = batch_rollout(material_model(known, 2.), future_states, future_probes, future_impacts, config["truth_dt"],
        config["target_times"], probe_time=config["probe_time"], future_time=config["future_time"], fracture_energies=future_energies)
    np.savez_compressed(root/"private/targets.npz", gaps=targets["opening"], failure=targets["damage"][:, -1] == 1.,
        x_left=targets["x_left"], x_right=targets["x_right"])
    write_json(root/"private/parameters.json", labels)
    write_json(root/"private/target-index.json", target_index)
    write_json(root/"public-manifest.json", {"schema": "joint-damage-material-public-1", "known_model": known,
        "times": config["context_times"], "probe_time": config["probe_time"], "noise": config["noise"],
        "parameter_bounds": config["parameter_bounds"], "cases": public})
    write_json(root/"inference-settings.json", {k: config[k] for k in ("methods", "budget", "optimizer", "acceptance", "inference_dt")})
    write_json(root/"future-plans.json", {"impacts": config["impacts"], "time": config["future_time"], "target_times": config["target_times"], "dt": config["inference_dt"]})
    print(f"JOINT_DATA_READY cases={len(public)} targets={len(target_index)}", flush=True)


class JointEngine:
    def __init__(self, root, manifest, settings):
        from cohesive_probe import BlockObservations
        self.root, self.public, self.settings = root, manifest, settings
        if set(manifest) != {"schema", "known_model", "times", "probe_time", "noise", "parameter_bounds", "cases"}:
            raise ValueError("Joint public input includes unexpected/privileged fields")
        if "fracture_energy" in manifest["known_model"]["law"]:
            raise ValueError("True material energy leaked into known model")
        self.observer = BlockObservations(manifest["known_model"])
        self.cases = {row["id"]: row for row in manifest["cases"]}
        self.observed, self.content_hashes, self.input_hashes = {}, {}, {}
        for case in manifest["cases"]:
            if set(case) != {"id", "probe", "rgbd_file"}:
                raise ValueError("Case input includes pair identities or true parameters")
            path = root/case["rgbd_file"]
            self.input_hashes[case["id"]] = digest(path)
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"rgb", "depth", "alpha"}:
                    raise ValueError("Privileged/future clip fields")
                arrays = {key: archive[key] for key in archive.files}
            if len(arrays["rgb"]) != len(manifest["times"]) or not all(np.isfinite(v).all() for v in arrays.values()):
                raise ValueError("Invalid observed prefix")
            self.observed[case["id"]] = [Observation(t, rgb, depth, alpha) for t, rgb, depth, alpha in zip(manifest["times"], arrays["rgb"], arrays["depth"], arrays["alpha"])]
            self.content_hashes[case["id"]] = hashlib.sha256(b"".join(arrays[k].tobytes() for k in ("rgb", "depth", "alpha"))).hexdigest()
        self.predictions, self.metrics, self.frames = {}, {}, OrderedDict()
        self.stats = {"material_history_replays": 0, "gaussian_renders": 0, "distinct_prefix_parameter_losses": 0}

    def predict(self, keys):
        fresh = list(dict.fromkeys(key for key in keys if key not in self.predictions))
        if not fresh:
            return
        releases = [reachable_state(material_model(self.public["known_model"], energy), peak).state for probe, peak, energy in fresh]
        values = batch_rollout(material_model(self.public["known_model"], 2.), releases, [k[0] for k in fresh], np.zeros(len(fresh)),
            self.settings["inference_dt"], self.public["times"], probe_time=self.public["probe_time"], fracture_energies=[k[2] for k in fresh])
        for row, key in enumerate(fresh):
            state = {k: (int(values[k][row, -1]) if k == "step_index" else float(values[k][row, -1])) for k in asdict(releases[row])}
            self.predictions[key] = {"state": state, "poses": np.stack([values["x_left"][row], values["x_right"][row]], axis=1)}
        self.stats["material_history_replays"] += len(fresh)

    def evaluate(self, identifier, unit):
        peak, energy = physical_parameters(unit, self.public["parameter_bounds"])
        case = self.cases[identifier]
        key = (case["probe"], peak, energy)
        score_key = (self.content_hashes[identifier], peak, energy)
        self.predict([key])
        if score_key not in self.metrics:
            if key in self.frames:
                frames = self.frames.pop(key)
            else:
                frames = [self.observer.capture({"x_left": x, "x_right": y}) for x, y in self.predictions[key]["poses"]]
                self.stats["gaussian_renders"] += len(frames)
            self.frames[key] = frames
            while len(self.frames) > 32:
                self.frames.popitem(last=False)
            residuals = [frame_residual(observed, rendered, self.public["noise"]) for observed, rendered in zip(self.observed[identifier], frames)]
            self.metrics[score_key] = {k: float(np.mean([r[k] for r in residuals])) for k in residuals[0]}
            self.stats["distinct_prefix_parameter_losses"] += 1
        return self.metrics[score_key], self.predictions[key]["state"]


def infer(root):
    start = time.perf_counter()
    manifest = json.loads((root/"public-manifest.json").read_text())
    settings = json.loads((root/"inference-settings.json").read_text())
    if set(settings) != {"methods", "budget", "optimizer", "acceptance", "inference_dt"}:
        raise ValueError("Joint inference settings include future or private fields")
    engine = JointEngine(root, manifest, settings)
    grid = [np.array([x, y]) for x in np.linspace(0, 1, 7) for y in np.linspace(0, 1, 7)]
    engine.predict([(probe, *physical_parameters(unit, manifest["parameter_bounds"]))
                    for probe in sorted({c["probe"] for c in manifest["cases"]}) for unit in grid])
    records = []
    for case in manifest["cases"]:
        methods = {}
        for method in settings["methods"]:
            began = time.perf_counter()
            def objective(x):
                return engine.evaluate(case["id"], x)[0]["score"]
            if method == "uniform49":
                run = {"samples": [x.tolist() for x in grid], "scores": [objective(x) for x in grid], "callback_evaluations": len(grid), "distinct_locations": len(grid)}
            elif method == "multiregion49":
                run = partition_search(objective, settings["budget"])
            else:
                run = run_direct(objective, [(0., 1.)]*2, settings["budget"], method == "directL49", settings["optimizer"])
            units = list(dict.fromkeys(tuple(row) for row in run["samples"]))
            parameters = [list(physical_parameters(unit, manifest["parameter_bounds"])) for unit in units]
            evaluated = [engine.evaluate(case["id"], unit) for unit in units]
            metrics = [item[0] for item in evaluated]
            accepted = accepted_indices(metrics, settings["acceptance"])
            representative = min(accepted, key=lambda i: (metrics[i]["score"], parameters[i])) if accepted else None
            methods[method] = {"parameters": parameters, "metrics": metrics, "cutoff_states": [item[1] for item in evaluated],
                "accepted_indices": accepted, "representative_index": representative, "uniquely_identified": False,
                "budget_audit": run, "wall_seconds_with_shared_cache": time.perf_counter()-began}
        records.append({"id": case["id"], "probe": case["probe"], "methods": methods})
        print(f"JOINT_CASE {len(records)}/{len(manifest['cases'])}", flush=True)
    out = root/"fitted"
    out.mkdir()
    packet = {"schema": "joint-reachable-state-material-1", "known_model": manifest["known_model"], "cases": records,
        "observation_end": max(manifest["times"]), "input_hashes": engine.input_hashes, "manifest_sha256": digest(root/"public-manifest.json"),
        "settings_sha256": digest(root/"inference-settings.json"), "wall_seconds": time.perf_counter()-start,
        "cache_statistics": engine.stats, "future_plans_read": False, "private_parameters_or_responses_read": False,
        "pretrained_or_neural_model_used": False}
    write_json(out/"selection.json", packet)
    (out/"selection.sha256").write_text(digest(out/"selection.json"))
    print("JOINT_SEALED", flush=True)


def sealed(root):
    path = root/"fitted/selection.json"
    if digest(path) != (root/"fitted/selection.sha256").read_text():
        raise ValueError("Joint material/state packet changed")
    return json.loads(path.read_text())


def forecast(root):
    packet = sealed(root)
    plans = json.loads((root/"future-plans.json").read_text())
    states, energies, impacts, keys, lookup = [], [], [], [], {}
    references = []
    for case in packet["cases"]:
        for method, record in case["methods"].items():
            for impact in plans["impacts"]:
                candidates = []
                for parameters, state in zip(record["parameters"], record["cutoff_states"]):
                    key = (case["probe"], *parameters, impact)
                    if key not in lookup:
                        lookup[key] = len(states); keys.append(key)
                        states.append(InterfaceState(**state)); energies.append(parameters[1]); impacts.append(impact)
                    candidates.append(lookup[key])
                references.append({"case": case["id"], "probe": case["probe"], "method": method, "impact": impact,
                    "candidate_rows": candidates, "accepted_indices": record["accepted_indices"], "representative_index": record["representative_index"]})
    values = batch_rollout(material_model(packet["known_model"], 2.), states, np.zeros(len(states)), impacts, plans["dt"],
        plans["target_times"], future_time=plans["time"], start_elapsed=packet["observation_end"], fracture_energies=energies)
    out = root/"forecast"
    out.mkdir()
    np.savez_compressed(out/"branches.npz", gaps=values["opening"], failure=values["damage"][:, -1] == 1., x_left=values["x_left"], x_right=values["x_right"])
    write_json(out/"index.json", references)
    audit = {"selection_sha256": digest(root/"fitted/selection.json"), "plans_sha256": digest(root/"future-plans.json"),
        "branches_sha256": digest(out/"branches.npz"), "index_sha256": digest(out/"index.json"),
        "unique_physical_branches": len(states), "candidate_material_and_state_preserved": True, "private_labels_read": False}
    write_json(out/"seal.json", audit)
    (out/"seal.sha256").write_text(digest(out/"seal.json"))
    print(f"JOINT_FORECAST branches={len(states)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--stage", choices=["generate", "infer", "forecast", "evaluate", "render", "branch"])
    parser.add_argument("--case")
    parser.add_argument("--method", default="directL49")
    parser.add_argument("--impact", type=float)
    parser.add_argument("--impact-time", type=float, default=.036)
    args = parser.parse_args()
    if args.stage == "branch":
        if args.case is None or args.impact is None:
            parser.error("branch requires --case and --impact")
        branch(args.run, args.case, args.method, args.impact, args.impact_time); return
    if args.stage in ("generate", "infer", "forecast"):
        globals()[args.stage](args.run); return
    if args.stage:
        from joint_damage_material_evaluate import evaluate, render
        {"evaluate": evaluate, "render": render}[args.stage](args.run); return
    project = Path(os.environ["WM_PROJECT_ROOT"])
    config = json.loads((project/"config/joint-damage-material.json").read_text())
    root = Path(os.environ["WM_RESEARCH_ROOT"])/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-joint-damage-material")
    root.mkdir(parents=True)
    write_json(root/"config.json", config)
    write_json(root/"code-sha256.json", {str(p.relative_to(project)): digest(p) for p in [Path(__file__), project/"wmfracture/joint_state.py",
        project/"wmfracture/budgeted_direct.py", project/"wmfracture/cohesive_batch.py", project/"scripts/joint_damage_material_evaluate.py"]})
    print(f"OUTPUT={root}", flush=True)
    for stage in ("generate", "infer", "forecast", "evaluate", "render"):
        launch([sys.executable, "-u", str(Path(__file__)), "--stage", stage, "--run", str(root)], root/f"{stage}.log")


def branch(root, identifier, method, impact, impact_time):
    packet = sealed(root)
    if not np.isfinite([impact, impact_time]).all() or impact_time <= packet["observation_end"]:
        raise ValueError("New impulse must be finite and after the sealed observation cutoff")
    case = next((case for case in packet["cases"] if case["id"] == identifier), None)
    if case is None or method not in case["methods"]:
        raise ValueError("Unknown joint material-state packet")
    record = case["methods"][method]
    if not record["accepted_indices"]:
        raise ValueError("No observation-matched material-state hypotheses")
    plans = json.loads((root/"future-plans.json").read_text())
    if impact_time >= max(plans["target_times"]):
        raise ValueError("New impulse outside the forecast horizon")
    parameters = np.array(record["parameters"])
    states = [InterfaceState(**state) for state in record["cutoff_states"]]
    values = batch_rollout(material_model(packet["known_model"], 2.), states, np.zeros(len(states)), np.full(len(states), impact),
        plans["dt"], plans["target_times"], future_time=impact_time, start_elapsed=packet["observation_end"], fracture_energies=parameters[:, 1])
    out = root/"branches"
    out.mkdir(exist_ok=True)
    stem = f"{identifier}-{method}-dv-{impact:g}-t-{impact_time:g}"
    np.savez_compressed(out/f"{stem}.npz", parameters=parameters, gaps=values["opening"], failure=values["damage"][:, -1] == 1.,
        accepted_indices=np.array(record["accepted_indices"]))
    write_json(out/f"{stem}.json", {"selection_sha256": digest(root/"fitted/selection.json"), "case": identifier, "method": method,
        "impact": impact, "impact_time": impact_time, "representative_index": record["representative_index"],
        "uniquely_identified": False, "material_and_state_preserved": True, "hypotheses": len(states), "test_reference_available": False})
    print(f"JOINT_BRANCH {out/stem}", flush=True)


if __name__ == "__main__":
    main()
