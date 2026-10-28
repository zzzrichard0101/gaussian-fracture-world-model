"""Same-budget observation-only history refinement and sealed physical forecasts."""
import argparse
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from wmfracture.cohesive import InterfaceState, InterfaceLaw, TwoBlockSystem
from wmfracture.cohesive_batch import batch_rollout
from wmfracture.cohesive_inference import Observation, ObservationBundle, frame_residual, reachable_state, validate_public
from wmfracture.history_refinement import canonical_peak, initial_points, propose, selection, validate_search


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def public_inputs(root):
    config = json.loads((root/"config.json").read_text())
    experiment = json.loads((root/"experiment.json").read_text())
    index = json.loads((root/"prefix-index.json").read_text())
    model = json.loads((root/"known-model.json").read_text())
    write_json(root/"observation-manifest.json", {"schema": "reduced-rgbd-search-inputs-1", "model": model,
        "times": config["context_times"], "probe_time": config["probe_time"], "noise": config["sensor_noise"],
        "cases": [{"id": f"prefix-{r['index']:03d}", "split": r["split"], "probe": r["probe"],
                   "rgbd_file": f"prefixes/prefix-{r['index']:03d}.npz"} for r in index]})
    write_json(root/"inference-policy.json", {"methods": experiment["methods"], "clip_cache_size": experiment["prediction_clip_cache_size"]})
    write_json(root/"future-plans.json", {"impacts": experiment["new_impacts"], "time": config["future_time"],
        "target_times": config["target_times"], "dt": experiment["search"]["dt"]})


def generate(root):
    from action_adapter_data import generate as generate_data
    generate_data(root)
    public_inputs(root)


class ObservationEngine:
    def __init__(self, root, header, settings, cache_size):
        from cohesive_probe import BlockObservations
        self.root, self.header, self.settings = root, header, settings
        self.observer = BlockObservations(header["model"])
        self.clip_cache, self.predictions, self.book = OrderedDict(), {}, {}
        self.cache_size = cache_size
        self.cases = {case["id"]: case for case in header["cases"]}
        self.observed, self.input_hashes = {}, {}
        for case in header["cases"]:
            path = root/case["rgbd_file"]
            self.input_hashes[case["id"]] = digest(path)
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"rgb", "depth", "alpha"}:
                    raise ValueError("Observed clip contains unexpected/private fields")
                fields = {key: archive[key] for key in archive.files}
            if len(fields["rgb"]) != len(header["times"]):
                raise ValueError("Observed clip length differs from declared cutoff")
            frames = tuple(Observation(t, rgb, depth, alpha) for t, rgb, depth, alpha in
                zip(header["times"], fields["rgb"], fields["depth"], fields["alpha"]))
            if not all(np.isfinite(getattr(frame, key)).all() for frame in frames for key in ("rgb", "depth", "object_alpha")):
                raise ValueError("Nonfinite observation")
            public = {"schema": "cohesive-observations-1", "model": header["model"], "observation_times": header["times"],
                "observation_end": max(header["times"]), "observation_actions": ([{"id": "observed-probe", "time": header["probe_time"], "left": case["probe"], "right": 0.}] if case["probe"] else []),
                "sensor_noise": header["noise"]}
            validate_public(public)
            self.observed[case["id"]] = ObservationBundle(public, frames)
        self.stats = {"physical_history_replays": 0, "gaussian_observation_renders": 0, "distinct_case_loss_evaluations": 0}

    def fulfil(self, requests):
        requested = {}
        for identifier, peak in requests:
            peak = canonical_peak(peak)
            if (identifier, peak) not in self.book:
                key = (self.cases[identifier]["probe"], peak)
                requested.setdefault(key, set()).add(identifier)
        new = [key for key in requested if key not in self.predictions]
        if new:
            releases = [reachable_state(self.header["model"], peak).state for probe, peak in new]
            result = batch_rollout(self.header["model"], releases, [key[0] for key in new], np.zeros(len(new)),
                self.settings["dt"], self.header["times"], probe_time=self.header["probe_time"])
            for row, key in enumerate(new):
                state = {field: (int(result[field][row, -1]) if field == "step_index" else float(result[field][row, -1])) for field in asdict(releases[row])}
                TwoBlockSystem(InterfaceLaw(**self.header["model"]["law"]), InterfaceState(**state),
                    self.header["model"]["mass_left"], self.header["model"]["mass_right"], self.header["model"]["width"])
                self.predictions[key] = {"state": state, "poses": np.stack([result["x_left"][row], result["x_right"][row]], axis=1)}
            self.stats["physical_history_replays"] += len(new)
        for key, identifiers in requested.items():
            if key in self.clip_cache:
                clip = self.clip_cache.pop(key)
            else:
                clip = [self.observer.capture({"x_left": x, "x_right": y}) for x, y in self.predictions[key]["poses"]]
                self.stats["gaussian_observation_renders"] += len(clip)
            self.clip_cache[key] = clip
            while len(self.clip_cache) > self.cache_size:
                self.clip_cache.popitem(last=False)
            for identifier in identifiers:
                residuals = [frame_residual(observed, generated, self.header["noise"])
                    for observed, generated in zip(self.observed[identifier].frames, clip)]
                self.book[(identifier, key[1])] = {field: float(np.mean([row[field] for row in residuals])) for field in residuals[0]}
                self.stats["distinct_case_loss_evaluations"] += 1


def infer(root):
    import torch
    start = time.perf_counter()
    header = json.loads((root/"observation-manifest.json").read_text())
    if set(header) != {"schema", "model", "times", "probe_time", "noise", "cases"}:
        raise ValueError("Public observation manifest includes unauthorized fields")
    settings = json.loads((root/"search-settings.json").read_text())
    validate_search(settings)
    policy = json.loads((root/"inference-policy.json").read_text())
    if set(policy) != {"methods", "clip_cache_size"}:
        raise ValueError("Inference policy includes future information")
    engine = ObservationEngine(root, header, settings, policy["clip_cache_size"])
    torch.cuda.reset_peak_memory_stats()
    searched, traces = {}, {}
    for case in header["cases"]:
        for method in policy["methods"]:
            key = (case["id"], method)
            searched[key] = initial_points(method, settings)
            traces[key] = []
    engine.fulfil([(identifier, peak) for (identifier, method), peaks in searched.items() for peak in peaks])
    rounds = (settings["budget"]-settings["initial_count"]+settings["batch_size"]-1)//settings["batch_size"]
    for round_index in range(rounds):
        requests = []
        for key, peaks in searched.items():
            identifier, method = key
            if method not in ("greedy49", "multiregion49"):
                continue
            extra = propose(peaks, [engine.book[(identifier, peak)]["score"] for peak in peaks], method, settings)
            traces[key].append({"round": round_index+1, "new_peaks": extra})
            requests.extend((identifier, peak) for peak in extra)
            peaks.extend(extra)
        engine.fulfil(requests)
        print(f"REFINEMENT_ROUND {round_index+1}/{rounds} logical_new_queries={len(requests)} cache_states={len(engine.predictions)}", flush=True)
    records = []
    for case in header["cases"]:
        methods = {}
        for method in policy["methods"]:
            peaks = sorted(searched[(case["id"], method)])
            expected = settings["initial_count"] if method == "coarse13" else settings["budget"]
            if len(peaks) != expected or len(set(peaks)) != expected:
                raise ValueError("Inference violated its distinct evaluation budget")
            metrics = [engine.book[(case["id"], peak)] for peak in peaks]
            chosen = selection(metrics, settings)
            methods[method] = {**chosen, "peaks": peaks, "metrics": metrics,
                "cutoff_states": [engine.predictions[(case["probe"], peak)]["state"] for peak in peaks],
                "logical_history_evaluations": len(peaks), "logical_observation_renders": len(peaks)*len(header["times"]),
                "proposal_trace": traces[(case["id"], method)]}
        records.append({"id": case["id"], "split": case["split"], "probe": case["probe"], "methods": methods})
    output = root/"fitted"
    output.mkdir()
    report = {"schema": "adaptive-reachable-history-packet-1", "cases": records, "model": header["model"],
        "observation_end": max(header["times"]), "public_observation_manifest_sha256": digest(root/"observation-manifest.json"),
        "observed_clip_hashes": engine.input_hashes, "search_settings_sha256": digest(root/"search-settings.json"),
        "inference_policy_sha256": digest(root/"inference-policy.json"), "cache_statistics": engine.stats,
        "wall_seconds": time.perf_counter()-start, "peak_torch_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
        "future_plan_read": False, "private_labels_read": False, "pretrained_or_neural_module_used": False}
    write_json(output/"selection.json", report)
    (output/"selection.sha256").write_text(digest(output/"selection.json"))
    print(f"REFINEMENT_SEALED cases={len(records)} seconds={report['wall_seconds']:.1f}", flush=True)


def sealed(root):
    path = root/"fitted/selection.json"
    if digest(path) != (root/"fitted/selection.sha256").read_text():
        raise ValueError("Fitted cutoff states changed after selection")
    return json.loads(path.read_text())


def forecast(root):
    chosen = sealed(root)
    plan = json.loads((root/"future-plans.json").read_text())
    states, impacts, index = [], [], []
    for case in chosen["cases"]:
        if case["split"] != "test":
            continue
        for method, record in case["methods"].items():
            selected = record["point_index"]
            if selected is None:
                continue
            state = InterfaceState(**record["cutoff_states"][selected])
            for impact in plan["impacts"]:
                states.append(InterfaceState(**asdict(state))); impacts.append(impact)
                index.append({"case": case["id"], "method": method, "impact": impact, "point_peak": record["peaks"][selected]})
    if not states:
        raise ValueError("No identifiable matched point states; refusing unique forecasts")
    values = batch_rollout(chosen["model"], states, np.zeros(len(states)), impacts, plan["dt"], plan["target_times"],
        future_time=plan["time"], start_elapsed=chosen["observation_end"])
    output = root/"forecast"
    output.mkdir()
    np.savez_compressed(output/"point-forecasts.npz", gaps=values["opening"], failure=values["damage"][:, -1] == 1.,
        x_left=values["x_left"], x_right=values["x_right"])
    write_json(output/"index.json", index)
    audit = {"selection_sha256": digest(root/"fitted/selection.json"), "future_plan_sha256": digest(root/"future-plans.json"),
        "forecast_sha256": digest(output/"point-forecasts.npz"), "index_sha256": digest(output/"index.json"),
        "states_not_reset_or_averaged": True, "test_labels_read": False, "branches": len(index)}
    write_json(output/"seal.json", audit)
    (output/"seal.sha256").write_text(digest(output/"seal.json"))
    print(f"REFINEMENT_FORECAST branches={len(index)}", flush=True)


def prepare(root, experiment):
    project = Path(os.environ["WM_PROJECT_ROOT"])
    data = json.loads((project/experiment["base_data_config"]).read_text())
    data.update({key: experiment[key] for key in ("seed", "history_counts", "history_peak_min_m", "history_peak_max_m")})
    data["actions"] = {"train": [], "validation": experiment["new_impacts"], "test_seen": experiment["new_impacts"], "test_interpolation": [], "test_extrapolation": []}
    data["write_belief_data"] = False
    data["scope"] = experiment["scope"]
    write_json(root/"config.json", data)
    write_json(root/"experiment.json", experiment)
    write_json(root/"search-settings.json", experiment["search"])


def branch(root, identifier, method, impact, impact_time):
    packet = sealed(root)
    if not np.isfinite([impact, impact_time]).all() or impact_time <= packet["observation_end"]:
        raise ValueError("New impulse must be finite and after the sealed observation cutoff")
    case = next((case for case in packet["cases"] if case["id"] == identifier), None)
    if case is None or method not in case["methods"]:
        raise ValueError("Unknown sealed case or search method")
    record = case["methods"][method]
    if not record["accepted_indices"]:
        raise ValueError("No observation-matched hypotheses; refusing a forecast")
    plan = json.loads((root/"future-plans.json").read_text())
    if impact_time >= max(plan["target_times"]):
        raise ValueError("New impulse outside the target horizon")
    states = [InterfaceState(**state) for state in record["cutoff_states"]]
    values = batch_rollout(packet["model"], states, np.zeros(len(states)), np.full(len(states), impact), plan["dt"],
        plan["target_times"], future_time=impact_time, start_elapsed=packet["observation_end"])
    output = root/"branches"
    output.mkdir(exist_ok=True)
    stem = f"{identifier}-{method}-dv-{impact:g}-t-{impact_time:g}"
    np.savez_compressed(output/f"{stem}.npz", gaps=values["opening"], failure=values["damage"][:, -1] == 1,
        peaks=np.array(record["peaks"]), accepted_indices=np.array(record["accepted_indices"]))
    write_json(output/f"{stem}.json", {"selection_sha256": digest(root/"fitted/selection.json"), "case": identifier,
        "method": method, "impact": impact, "impact_time": impact_time, "point_index": record["point_index"],
        "hypotheses": len(states), "observation_matched_hypotheses": len(record["accepted_indices"]),
        "point_forecast_available": record["point_index"] is not None,
        "states_not_reset_or_averaged": True, "test_reference_available": False,
        "note": "Retained alternatives are physical hypotheses, not a calibrated forecast distribution."})
    print(f"REFINEMENT_BRANCH {output/stem} hypotheses={len(states)}", flush=True)


def launch(command, log):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in process.stdout:
            stream.write(line); stream.flush(); tail = (tail+[line])[-20:]
            if line.startswith(("PREFIXES", "DATA_READY", "REFINEMENT", "GAUSSIAN_VIDEO", "DIRECT", "JOINT")):
                print(line, end="", flush=True)
        if process.wait():
            print("".join(tail), flush=True)
            raise subprocess.CalledProcessError(process.returncode, command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/history-refinement.json"))
    parser.add_argument("--stage", choices=["generate", "infer", "forecast", "evaluate", "render", "branch", "continue"])
    parser.add_argument("--case")
    parser.add_argument("--method", default="multiregion49")
    parser.add_argument("--impact", type=float)
    parser.add_argument("--impact-time", type=float, default=.036)
    args = parser.parse_args()
    if args.stage == "branch":
        if args.case is None or args.impact is None:
            parser.error("branch requires --case and --impact")
        branch(args.run, args.case, args.method, args.impact, args.impact_time); return
    if args.stage in ("generate", "infer", "forecast"):
        globals()[args.stage](args.run); return
    if args.stage in ("evaluate", "render"):
        from history_refinement_evaluate import evaluate, render
        {"evaluate": evaluate, "render": render}[args.stage](args.run); return
    project = Path(os.environ["WM_PROJECT_ROOT"])
    if args.run:
        root = args.run
        public_inputs(root)
    else:
        experiment = json.loads((project/args.config).read_text())
        root = Path(os.environ["WM_RESEARCH_ROOT"])/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-history-refinement")
        root.mkdir(parents=True)
        prepare(root, experiment)
        print(f"OUTPUT={root}", flush=True)
        launch([sys.executable, "-u", str(Path(__file__)), "--stage", "generate", "--run", str(root)], root/"generate.log")
    write_json(root/"code-sha256.json", {str(path.relative_to(project)): digest(path) for path in [Path(__file__),
        project/"scripts/history_refinement_evaluate.py", project/"wmfracture/history_refinement.py",
        project/"wmfracture/cohesive_batch.py", project/"wmfracture/cohesive.py", project/"wmfracture/cohesive_inference.py"]})
    for stage in ("infer", "forecast", "evaluate", "render"):
        launch([sys.executable, "-u", str(Path(__file__)), "--stage", stage, "--run", str(root)], root/f"{stage}.log")


if __name__ == "__main__":
    main()
