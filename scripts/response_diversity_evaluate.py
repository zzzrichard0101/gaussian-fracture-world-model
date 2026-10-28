"""Read private targets only after physical pool and compression are frozen."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from history_refinement import digest, write_json
from response_diversity import read_sealed, read_bank
from wmfracture.fixture_reference import reference_parameters


def evaluate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pool = read_sealed(root/"fitted/pool.json")
    selection = read_sealed(root/"compressed/selection.json")
    config = json.loads((root/"config.json").read_text())
    assert digest(root/"fitted/pool.json") == selection["pool_sha256"]
    assert digest(root/"design/seal.json") == selection["design_seal_sha256"]
    assert digest(root/"compression-settings.json") == selection["settings_sha256"]
    seal = read_sealed(root/"forecast/seal.json")
    assert seal["compression_sha256"] == digest(root/"compressed/selection.json")
    for folder, path in (("design", "design-plans.json"), ("forecast", "future-plans.json")):
        assert read_sealed(root/folder/"seal.json")["plans_sha256"] == digest(root/path)
    index, values = read_bank(root, "forecast")
    # The first private reads of material labels and future responses in this pipeline.
    labels = json.loads((root/"private/parameters.json").read_text())
    targets = json.loads((root/"private/target-index.json").read_text())
    with np.load(root/"private/targets.npz", allow_pickle=False) as archive:
        truth = {k: archive[k] for k in archive.files}
    target_lookup = {(r["case"], r["impact"]): i for i, r in enumerate(targets)}
    lookup = {(r["case"], r["pool"]): np.array(r["rows"]) for r in index}
    record_lookup = {r["id"]: r for r in pool["cases"]}
    rows = []
    for case in selection["cases"]:
        for name, rules in case["pools"].items():
            source = record_lookup[case["id"]]["pools"][name]
            matrix = lookup[(case["id"], name)]
            all_accepted = source["accepted_indices"]
            for rule, selected in rules.items():
                assert set(selected) <= set(all_accepted) and len(set(selected)) == len(selected)
                assert rule == "full" or len(selected) <= config["compression"]["count"]
                for a, action in enumerate(config["impacts"]):
                    base = {"case": case["id"], "pool": name, "rule": rule, "probe": case["probe"],
                        "impact": action, "states_retained": len(selected), "prediction_available": bool(selected)}
                    if not selected:
                        rows.append(base); continue
                    target = target_lookup[(case["id"], action)]
                    curves = values["gaps"][matrix[selected, a]]
                    events = values["failure"][matrix[selected, a]]
                    reference = values["gaps"][matrix[all_accepted, a]]
                    lower, upper = curves.min(0), curves.max(0)
                    outside = np.maximum(lower-truth["gaps"][target], 0)+np.maximum(truth["gaps"][target]-upper, 0)
                    distances = np.sqrt(np.mean((reference[:, None, :]-curves[None, :, :])**2, axis=-1))
                    rows.append({**base,
                        "nearest_candidate_mse_m2": float(np.min(np.mean((curves-truth["gaps"][target])**2, axis=1))),
                        "envelope_time_coverage": float(np.mean(outside == 0)),
                        "envelope_width_mm": float(np.mean(upper-lower)*1000),
                        "outside_distance_mm": float(np.mean(outside)*1000),
                        "width_plus_missing_penalty_mm": float(np.mean(upper-lower+config["compression"]["outside_distance_penalty"]*outside)*1000),
                        "event_set_contains_truth": bool(np.any(events == truth["failure"][target])),
                        "event_set_ambiguous": bool(events.min() != events.max()),
                        "full_pool_response_max_nearest_rmse_mm": float(distances.min(axis=1).max()*1000)})
    summaries = {}
    for probe in config["probes"]:
        summaries[str(probe)] = {}
        for name in config["methods"]:
            summaries[str(probe)][name] = {}
            for rule in selection["cases"][0]["pools"][name]:
                selected = [r for r in rows if r["probe"] == probe and r["pool"] == name and r["rule"] == rule]
                good = [r for r in selected if r["prediction_available"]]
                summary = {"forecast_cases": len(selected), "unavailable_forecasts": len(selected)-len(good)}
                if good:
                    summary["nearest_candidate_rmse_mm"] = float(np.sqrt(np.mean([r["nearest_candidate_mse_m2"] for r in good]))*1000)
                    for key in ("states_retained", "envelope_time_coverage", "envelope_width_mm", "outside_distance_mm",
                                "width_plus_missing_penalty_mm", "event_set_contains_truth", "event_set_ambiguous", "full_pool_response_max_nearest_rmse_mm"):
                        summary[key] = float(np.mean([r[key] for r in good]))
                summaries[str(probe)][name][rule] = summary
    # Check new true materials/history peaks against all previous joint-control states.
    previous = Path(os.environ["WM_PROJECT_ROOT"])/"config/joint-damage-material.json"
    old_labels = reference_parameters(previous, pool["known_model"])
    energies = {round(r["energy"], 12) for r in labels.values()}
    peaks = {round(r["peak"], 15) for r in labels.values()}
    disjoint_energy = energies.isdisjoint({round(r["energy"], 12) for r in old_labels})
    disjoint_peaks = peaks.isdisjoint({round(r["peak"], 15) for r in old_labels})
    assert disjoint_energy and disjoint_peaks
    assert not set(config["design_impacts"]) & set(config["impacts"])
    audits = {stage: json.loads((root/f"{stage}-read-audit.json").read_text()) for stage in ("infer", "design", "compress", "forecast")}
    assert all(not path.startswith("private/") and path != "config.json" for paths in audits.values() for path in paths)
    assert "future-plans.json" not in audits["compress"] and "design-plans.json" not in audits["infer"]
    assert all(r["budget_audit"]["callback_evaluations"] == 49 for c in pool["cases"] for r in c["pools"].values())
    for name, expected in pool["input_hashes"].items():
        assert digest(root/"public"/(name+".npz")) == expected
    controls = []
    for probe in config["probes"]:
        for pair in range(config["pairs"]):
            pair_cases = [case for case, label in labels.items() if label["pair"] == pair and label["probe"] == probe]
            clips = []
            for case in pair_cases:
                with np.load(root/"public"/(case+".npz"), allow_pickle=False) as archive:
                    clips.append({k: archive[k] for k in archive.files})
            equal = all(np.array_equal(clips[0][k], clips[1][k]) for k in clips[0])
            if equal:
                left, right = [next(c for c in selection["cases"] if c["id"] == name) for name in pair_cases]
                assert left["pools"] == right["pools"]
            controls.append({"pair": pair, "probe": probe, "observed_arrays_identical": equal})
    # Demo fixed to first public case, uniform pool, middle held-out impact.
    first = selection["cases"][0]
    source_rows = lookup[(first["id"], "uniform49")]
    action = config["impacts"][1]; a = 1
    target = target_lookup[(first["id"], action)]
    demo = {"truth_gap": truth["gaps"][target], "truth_xleft": truth["x_left"][target], "truth_xright": truth["x_right"][target]}
    for rule in ("full", "residual8", "parameter8", "response8"):
        selected = first["pools"]["uniform49"][rule]
        demo[rule] = values["gaps"][source_rows[selected, a]]
        demo[rule+"_xleft"] = values["x_left"][source_rows[selected, a]]
        demo[rule+"_xright"] = values["x_right"][source_rows[selected, a]]
    np.savez_compressed(root/"demo.npz", **demo)
    write_json(root/"demo.json", {"case": first["id"], "pool": "uniform49", "impact": action,
        "rule": "First public case, uniform pool, middle test action; not selected by accuracy"})
    rules = ["residual8", "parameter8", "response8", "random8-17", "random8-37", "random8-59", "full"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for p, probe in enumerate(config["probes"]):
        group = summaries[str(probe)]["uniform49"]
        axes[0].bar(np.arange(len(rules))+(p-.5)*.35, [group[r]["nearest_candidate_rmse_mm"] for r in rules], .35, label=f"Probe {probe}m/s")
    axes[0].set_xticks(np.arange(len(rules)), rules, rotation=45, ha="right")
    axes[0].set_ylabel("Nearest candidate to truth (mm); set diagnostic")
    for name in config["methods"]:
        xs = [summaries[str(p)][name]["parameter8"]["full_pool_response_max_nearest_rmse_mm"] for p in config["probes"]]
        ys = [summaries[str(p)][name]["response8"]["full_pool_response_max_nearest_rmse_mm"] for p in config["probes"]]
        axes[1].scatter(xs, ys, label=name)
    maximum = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1]); axes[1].plot([0, maximum], [0, maximum], "k--", alpha=.4)
    axes[1].set_xlabel("Parameter8 full-pool representation error (mm)")
    axes[1].set_ylabel("Response8 full-pool representation error (mm)")
    for rule in ("residual8", "parameter8", "response8"):
        curves = demo[rule]*1000
        axes[2].fill_between(config["target_times"], curves.min(0), curves.max(0), alpha=.16, label=rule)
    axes[2].plot(config["target_times"], demo["truth_gap"]*1000, "k", label="Truth (evaluation only)")
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("First case candidate ranges (mm)")
    for axis in axes:
        axis.legend(fontsize=6); axis.grid(alpha=.2)
    fig.savefig(root/"compression-comparison.png", dpi=150); plt.close(fig)
    report = {"status": "physical_hypothesis_compression_completed", "run_directory": str(root), "summaries": summaries,
        "per_forecast_results": rows, "observation_pair_controls": controls, "stage_input_audits": audits,
        "checks": {"true_materials_and_histories_disjoint_from_previous_joint_fixture": True,
            "design_and_test_actions_disjoint": True, "private_and_parent_config_reads_blocked_before_evaluation": True,
            "full_pool_frozen_before_design_actions": True, "compression_frozen_before_test_actions": True,
            "all_pool_callback_evaluations_equal_49": True, "all_compressed_sets_are_subsets_of_accepted_physical_states": True,
            "original_observation_inputs_unchanged": True},
        "budget": {"observation_callbacks_per_pool_case": 49, "design_trajectory_allowance_per_pool_case": 147,
            "compressed_states_maximum": 8, "full_reference_states_maximum": 49,
            "design_actual_unique_physical_branches": read_sealed(root/"design/seal.json")["unique_physical_branches"],
            "test_reference_actual_unique_physical_branches": seal["unique_physical_branches"]},
        "pool_inference": {k: pool[k] for k in ("seconds", "cache_statistics")},
        "new_method_novelty_validated": False, "pretrained_or_neural_model_used": False,
        "limitations": ["Farthest-first k-center compression is standard; pipeline utility does not establish novelty.",
            "Nearest-candidate error uses true futures only for evaluation, and cannot be deployed as a point predictor.",
            "Envelope width plus 20 times outside distance is a preset engineering score, not a calibrated interval score or posterior.",
            "Full accepted pool is an upper-budget reference, not an eight-state competitor. All design trajectories are charged even when a selector ignores them.",
            "Test reference computes every pool trajectory for diagnostics; no measured end-to-end speedup or uncomputed-event guarantee.",
            "Constructed matched-stiffness pairs only; not an independent real-object or general 3D benchmark. No active probe policy yet."]}
    write_json(root/"report.json", report)
    out = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/response-diversity"; out.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "compression-comparison.png", "config.json", "demo.json", "code-sha256.json"):
        shutil.copy2(root/name, out/name)
    print("DIVERSITY_EVALUATED "+json.dumps({"uniform49": {str(p): summaries[str(p)]["uniform49"] for p in config["probes"]}, "checks": report["checks"]}), flush=True)


def render(root):
    from cohesive_probe import BlockObservations
    from PIL import Image, ImageDraw
    pool = read_sealed(root/"fitted/pool.json")
    config = json.loads((root/"config.json").read_text())
    demo = json.loads((root/"demo.json").read_text())
    with np.load(root/"demo.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    observer = BlockObservations(pool["known_model"])
    width = config["render_width"]
    frames = root/"video-frames"; frames.mkdir(exist_ok=True)
    # Show both gap extremes of response8, not the evaluation-nearest candidate.
    order = np.argsort(data["response8"][:, -1])
    for t, elapsed in enumerate(config["target_times"]):
        canvas = Image.new("RGB", (width*3, width+64), "white"); draw = ImageDraw.Draw(canvas)
        poses = [("True held-out response", data["truth_xleft"][t], data["truth_xright"][t], data["truth_gap"][t])]
        for title, row in (("Response8: smaller final gap", order[0]), ("Response8: larger final gap", order[-1])):
            poses.append((title, data["response8_xleft"][row, t], data["response8_xright"][row, t], data["response8"][row, t]))
        for column, (title, left, right, gap) in enumerate(poses):
            rgb = observer.capture({"x_left": left, "x_right": right})["rgb"]
            canvas.paste(Image.fromarray(np.round(np.clip(rgb, 0, 1)*255).astype(np.uint8)), (column*width, 0))
            draw.text((column*width+5, width+4), title, fill="black")
            draw.text((column*width+5, width+21), f"gap={gap*1000:.2f}mm", fill="black")
        draw.text((5, width+43), f"t={elapsed:.3f}s | new dv={demo['impact']:.3f}m/s | distinct physical hypotheses", fill="black")
        canvas.save(frames/f"{t:05d}.png")
    canvas.save(root/"prediction-end.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i", str(frames/"%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(root/"compressed-futures.mp4")], check=True)
    out = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/response-diversity"
    for name in ("prediction-end.png", "compressed-futures.mp4"):
        shutil.copy2(root/name, out/name)
    print(f"DIVERSITY_VIDEO={root/'compressed-futures.mp4'}", flush=True)
