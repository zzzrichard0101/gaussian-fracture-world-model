"""Action-specific evaluation after closed-loop prediction has been sealed."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from history_refinement import digest, write_json
from response_diversity import read_sealed
from wmfracture.fixture_reference import reference_parameters


def evaluate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    config = json.loads((root/"config.json").read_text())
    pool = read_sealed(root/"fitted/pool.json")
    compressed = read_sealed(root/"compressed/selection.json")
    decision = read_sealed(root/"planning/decision.json")
    observed = read_sealed(root/"new-observations.json")
    updated = read_sealed(root/"updated/state.json")
    seal = read_sealed(root/"active-forecast/seal.json")
    for actual, expected in ((digest(root/"fitted/pool.json"), decision["pool_sha256"]),
        (digest(root/"compressed/selection.json"), decision["compression_sha256"]),
        (digest(root/"planning/decision.json"), observed["decision_sha256"]),
        (digest(root/"planning/decision.json"), updated["decision_sha256"]),
        (digest(root/"new-observations.json"), updated["observations_sha256"]),
        (digest(root/"updated/state.json"), seal["updated_sha256"]),
        (digest(root/"future-plans.json"), seal["plans_sha256"]),
        (digest(root/"active-forecast/bank.npz"), seal["bank_sha256"]),
        (digest(root/"active-forecast/index.json"), seal["index_sha256"]),
        (digest(root/"planning/bank.npz"), decision["bank_sha256"]),
        (digest(root/"planning/index.json"), decision["index_sha256"]),
        (digest(root/"probe-settings.json"), decision["settings_sha256"])):
        assert actual == expected
    index = json.loads((root/"active-forecast/index.json").read_text())
    with np.load(root/"active-forecast/bank.npz", allow_pickle=False) as archive:
        prediction = {k: archive[k] for k in archive.files}
    # Trusted environment used these earlier; this is the evaluator's first private read.
    labels = json.loads((root/"private/parameters.json").read_text())
    costs = json.loads((root/"private/probe-costs.json").read_text())
    target_index = json.loads((root/"private/action-target-index.json").read_text())
    with np.load(root/"private/action-targets.npz", allow_pickle=False) as archive:
        truth = {k: archive[k] for k in archive.files}
    target_lookup = {(r["case"], r["action"], r["impact"]): i for i, r in enumerate(target_index)}
    cost_lookup = {(r["case"], r["action"]): r for r in costs}
    state_lookup = {(r["case"], r["pool"], r["state_set"], r["policy"]): r for r in updated["records"]}
    rows, episodes = [], []
    for key, state in state_lookup.items():
        actual = cost_lookup[(state["case"], state["selected_impulse"])]
        passive = cost_lookup[(state["case"], 0.)]
        added = max(0., actual["damage"]-passive["damage"])
        newly_failed = bool(actual["failure"] and not passive["failure"])
        predicted = state["max_candidate_added_damage"][state["selected_action_index"]]
        episodes.append({"case": key[0], "pool": key[1], "state_set": key[2], "policy": key[3],
            "selected_impulse": state["selected_impulse"], "prior_count": len(state["prior_indices"]),
            "posterior_count": len(state["posterior_local_indices"]), "unavailable": not state["posterior_local_indices"],
            "actual_added_damage": added, "actual_additional_separation": newly_failed,
            "actual_damage_cap_exceeded": added > config["active"]["damage_cap"]+1e-12,
            "predicted_max_added_damage": predicted,
            "candidate_guard_holds": bool(state["eligible_actions"][state["selected_action_index"]])})
    for record in index:
        target = target_lookup[(record["case"], record["probe_action"], record["impact"])]
        candidates = np.array(record["candidate_rows"])
        prior_point = candidates[record["before_point_local_index"]]
        before = prediction["gaps"][prior_point]
        row = {"case": record["case"], "pool": record["pool"], "state_set": record["state_set"], "policy": record["policy"],
            "probe_action": record["probe_action"], "impact": record["impact"],
            "before_map_mse_m2": float(np.mean((before-truth["gaps"][target])**2)),
            "before_map_separation_correct": bool(prediction["failure"][prior_point] == truth["failure"][target]),
            "after_available": record["after_point_local_index"] is not None}
        if row["after_available"]:
            point = candidates[record["after_point_local_index"]]
            retained = candidates[record["posterior_local_indices"]]
            curves = prediction["gaps"][retained]
            lower, upper = curves.min(0), curves.max(0)
            row.update({"after_map_mse_m2": float(np.mean((prediction["gaps"][point]-truth["gaps"][target])**2)),
                "after_map_separation_correct": bool(prediction["failure"][point] == truth["failure"][target]),
                "nearest_retained_mse_m2": float(np.min(np.mean((curves-truth["gaps"][target])**2, axis=1))),
                "envelope_width_mm": float(np.mean(upper-lower)*1000),
                "envelope_time_coverage": float(np.mean((truth["gaps"][target] >= lower)&(truth["gaps"][target] <= upper))),
                "event_set_contains_truth": bool((prediction["failure"][retained] == truth["failure"][target]).any())})
        rows.append(row)
    summaries = {}
    for pool_name in config["methods"]:
        summaries[pool_name] = {}
        for state_set in config["active"]["state_sets"]:
            summaries[pool_name][state_set] = {}
            for policy in config["active"]["policies"]:
                forecasts = [r for r in rows if (r["pool"], r["state_set"], r["policy"]) == (pool_name, state_set, policy)]
                eps = [r for r in episodes if (r["pool"], r["state_set"], r["policy"]) == (pool_name, state_set, policy)]
                matched = [r for r in forecasts if r["after_available"]]
                result = {"episodes": len(eps), "forecasts": len(forecasts), "unavailable_forecasts": len(forecasts)-len(matched),
                    "before_map_all_rmse_mm": float(np.sqrt(np.mean([r["before_map_mse_m2"] for r in forecasts]))*1000),
                    "actions": {str(action): sum(e["selected_impulse"] == action for e in eps) for action in config["active"]["actions"]}}
                for metric in ("prior_count", "posterior_count", "actual_added_damage", "actual_additional_separation", "actual_damage_cap_exceeded", "predicted_max_added_damage", "candidate_guard_holds"):
                    result[metric] = float(np.mean([r[metric] for r in eps]))
                if matched:
                    result["before_map_matched_rmse_mm"] = float(np.sqrt(np.mean([r["before_map_mse_m2"] for r in matched]))*1000)
                    result["after_map_matched_rmse_mm"] = float(np.sqrt(np.mean([r["after_map_mse_m2"] for r in matched]))*1000)
                    result["nearest_retained_rmse_mm"] = float(np.sqrt(np.mean([r["nearest_retained_mse_m2"] for r in matched]))*1000)
                    for metric in ("after_map_separation_correct", "envelope_width_mm", "envelope_time_coverage", "event_set_contains_truth"):
                        result[metric] = float(np.mean([r[metric] for r in matched]))
                summaries[pool_name][state_set][policy] = result
    # New material and h values must be disjoint from both previous 2D fixtures.
    previous_configs = ["config/joint-damage-material.json", "config/response-diversity.json"]
    for previous in previous_configs:
        previous_labels = reference_parameters(Path(os.environ["WM_PROJECT_ROOT"])/previous, pool["known_model"])
        for field, decimals in (("energy", 12), ("peak", 15)):
            assert {round(r[field], decimals) for r in labels.values()}.isdisjoint({round(r[field], decimals) for r in previous_labels})
    assert not set(config["impacts"]) & set(config["design_impacts"])
    audits = {stage: json.loads((root/f"{stage}-read-audit.json").read_text()) for stage in ("infer", "design", "compress", "plan", "update", "active-forecast")}
    for stage, files in audits.items():
        assert not any(p.startswith("private/") or p == "config.json" for p in files)
        if stage != "active-forecast":
            assert "future-plans.json" not in files
    assert not any(p.startswith("public-extra/") for p in audits["plan"])
    assert all(e["candidate_guard_holds"] for e in episodes if e["policy"] == "guarded")
    assert all(set(r["posterior_local_indices"]) <= set(range(len(r["prior_indices"]))) for r in updated["records"])
    for case, expected in pool["input_hashes"].items():
        assert digest(root/"public"/(case+".npz")) == expected
    # Fixed demo: first case, uniform49, response8, guarded, middle test impulse.
    demo_record = next(r for r in index if r["case"] == "case-000" and r["pool"] == "uniform49" and r["state_set"] == "response8" and r["policy"] == "guarded" and r["impact"] == config["impacts"][1])
    target = target_lookup[(demo_record["case"], demo_record["probe_action"], demo_record["impact"])]
    before = demo_record["candidate_rows"][demo_record["before_point_local_index"]]
    after = (demo_record["candidate_rows"][demo_record["after_point_local_index"]] if demo_record["after_point_local_index"] is not None else None)
    demo = {"truth": truth["gaps"][target], "truth_xleft": truth["x_left"][target], "truth_xright": truth["x_right"][target]}
    for name, row in (("before", before), ("after", after)):
        if row is not None:
            demo[name] = prediction["gaps"][row]; demo[name+"_xleft"] = prediction["x_left"][row]; demo[name+"_xright"] = prediction["x_right"][row]
    np.savez_compressed(root/"demo.npz", **demo)
    write_json(root/"demo.json", {"case": "case-000", "pool": "uniform49", "state_set": "response8", "policy": "guarded",
        "probe_action": demo_record["probe_action"], "future_impact": demo_record["impact"], "after_available": after is not None,
        "rule": "First case, uniform49 response8 guarded, middle test action; fixed before results"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    policies = config["active"]["policies"]
    for i, state_set in enumerate(config["active"]["state_sets"]):
        group = summaries["uniform49"][state_set]
        axes[0].bar(np.arange(len(policies))+(i-.5)*.36, [group[p].get("after_map_matched_rmse_mm", np.nan) for p in policies], .36, label=state_set)
        axes[1].bar(np.arange(len(policies))+(i-.5)*.36, [group[p]["actual_added_damage"] for p in policies], .36, label=state_set)
    axes[0].set_ylabel("After-observation MAP RMSE (mm), available only")
    axes[1].set_ylabel("Actual extra damage vs passive (0 to 1)")
    axes[1].axhline(config["active"]["damage_cap"], color="red", linestyle="--", label="Candidate cap")
    for axis in axes[:2]:
        axis.set_xticks(np.arange(len(policies)), policies, rotation=45, ha="right")
    for key in ("truth", "before", "after"):
        if key in demo:
            axes[2].plot(config["target_times"], demo[key]*1000, label=key)
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Same probe-specific target, gap (mm)")
    for axis in axes:
        axis.legend(fontsize=6); axis.grid(alpha=.2)
    fig.savefig(root/"active-comparison.png", dpi=150); plt.close(fig)
    report = {"status": "closed_loop_active_probe_completed", "run_directory": str(root), "summaries": summaries,
        "per_episode_results": episodes, "per_forecast_results": rows, "stage_input_audits": audits,
        "checks": {"new_materials_and_histories_disjoint_from_two_previous_fixtures": True,
            "policy_frozen_before_new_actual_observations": True, "update_and_policy_exclude_private_truth_and_future_test_plan": True,
            "targets_include_actual_probe_induced_state_changes": True, "all_guarded_choices_pass_their_own_finite_candidate_guard": True,
            "updated_hypotheses_are_subsets_of_original_states": True, "original_observation_inputs_unchanged": True},
        "budget": {"observation_callbacks_per_pool_case": 49, "design_allowance_per_pool_case": 147,
            "planning_action_grid": len(config["active"]["actions"]), "planning_unique_physical_trajectories": decision["unique_candidate_action_trajectories"],
            "planning_gaussian_frames": decision["gaussian_planning_frames"], "planning_seconds": decision["seconds"],
            "future_unique_physical_branches": seal["unique_physical_branches"], "additional_observation_windows_per_episode": 1},
        "limitations": ["MAP metric names mean the minimum merged-residual representative here; no normalized posterior or calibrated likelihood is claimed.",
            "Known-color Gaussian centroids/depth/area at heuristic scales, not calibrated information gain or arbitrary-object perception.",
            "Finite candidate damage constraint is not a true-state safety guarantee; report actual violations including zero if none occurred.",
            "Policies change the actual target dynamics. Only within-policy pre/post measurement compares the same target; cross-policy RMSE is descriptive.",
            "No optimized continuous re-identification; observations filter a fixed prior candidate set and may eliminate it entirely.",
            "Nearest-set metrics use truth only for evaluation. Candidate count reduction and coverage are not posterior calibration.",
            "Eight constructed true states reused across methods/actions/seeds. No independent real objects, 3D fracture or pretrained-model benefit.",
            "Task-weighted probing and finite-set risk constraints are conventional principles; novel research contribution not established."],
        "new_method_novelty_validated": False, "pretrained_or_neural_model_used": False, "true_state_safety_guaranteed": False}
    write_json(root/"report.json", report)
    out = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/active-probe"; out.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "config.json", "code-sha256.json", "active-comparison.png", "demo.json"):
        shutil.copy2(root/name, out/name)
    print("ACTIVE_EVALUATED "+json.dumps({"uniform49": summaries["uniform49"], "checks": report["checks"]}), flush=True)


def render(root):
    from cohesive_probe import BlockObservations
    from PIL import Image, ImageDraw
    pool = read_sealed(root/"fitted/pool.json")
    config = json.loads((root/"config.json").read_text())
    demo = json.loads((root/"demo.json").read_text())
    with np.load(root/"demo.npz", allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    observer = BlockObservations(pool["known_model"])
    width = config["render_width"]
    frames = root/"video-frames"; frames.mkdir(exist_ok=True)
    for t, elapsed in enumerate(config["target_times"]):
        canvas = Image.new("RGB", (width*3, width+64), "white"); draw = ImageDraw.Draw(canvas)
        for column, (name, title) in enumerate((("truth", "Truth after selected probe"), ("before", "Best residual before evidence"), ("after", "Best residual after evidence"))):
            if name in values:
                rgb = observer.capture({"x_left": values[name+"_xleft"][t], "x_right": values[name+"_xright"][t]})["rgb"]
                canvas.paste(Image.fromarray(np.round(np.clip(rgb, 0, 1)*255).astype(np.uint8)), (column*width, 0))
                draw.text((column*width+5, width+21), f"gap={values[name][t]*1000:.2f}mm", fill="black")
            else:
                draw.text((column*width+20, width//2), "No matched hypothesis", fill="black")
            draw.text((column*width+5, width+4), title, fill="black")
        draw.text((5, width+43), f"t={elapsed:.3f}s | selected probe={demo['probe_action']:.3f}m/s | future dv={demo['future_impact']:.3f}m/s", fill="black")
        canvas.save(frames/f"{t:05d}.png")
    canvas.save(root/"prediction-end.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i", str(frames/"%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(root/"active-probe-futures.mp4")], check=True)
    out = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/active-probe"
    for name in ("prediction-end.png", "active-probe-futures.mp4"):
        shutil.copy2(root/name, out/name)
    print(f"ACTIVE_VIDEO={root/'active-probe-futures.mp4'}", flush=True)
