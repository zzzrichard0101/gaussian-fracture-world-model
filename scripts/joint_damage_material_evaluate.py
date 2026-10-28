"""Private-label evaluation of unknown material/history and paired ambiguity."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from history_refinement import digest, write_json
from joint_damage_material import sealed


def read_forecasts(root):
    packet = sealed(root)
    seal = json.loads((root/"forecast/seal.json").read_text())
    if digest(root/"forecast/seal.json") != (root/"forecast/seal.sha256").read_text():
        raise ValueError("Joint forecast seal changed")
    for key, relative in [("selection_sha256", "fitted/selection.json"), ("plans_sha256", "future-plans.json"),
                          ("branches_sha256", "forecast/branches.npz"), ("index_sha256", "forecast/index.json")]:
        if digest(root/relative) != seal[key]:
            raise ValueError("Joint forecast dependency changed")
    with np.load(root/"forecast/branches.npz", allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    return packet, json.loads((root/"forecast/index.json").read_text()), values


def evaluate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    packet, index, values = read_forecasts(root)
    config = json.loads((root/"config.json").read_text())
    # First true material/history/response reads after frozen inference and forecasts.
    labels = json.loads((root/"private/parameters.json").read_text())
    target_index = json.loads((root/"private/target-index.json").read_text())
    with np.load(root/"private/targets.npz", allow_pickle=False) as archive:
        actual = {key: archive[key] for key in archive.files}
    target_lookup = {(r["case"], r["impact"]): i for i, r in enumerate(target_index)}
    prediction_lookup = {(r["case"], r["method"], r["impact"]): r for r in index}
    case_lookup = {r["id"]: r for r in packet["cases"]}
    pair_audit, floors = [], {}
    for probe in config["probes"]:
        squared_differences = []
        for pair in range(config["pairs"]):
            members = {label["member"]: identifier for identifier, label in labels.items() if label["pair"] == pair and label["probe"] == probe}
            observed = []
            for member in ("low", "high"):
                with np.load(root/"public"/f"{members[member]}.npz", allow_pickle=False) as archive:
                    observed.append({key: archive[key] for key in archive.files})
            equal = all(np.array_equal(observed[0][key], observed[1][key]) for key in observed[0])
            differences, event_disagreements = [], 0
            representative_changes = {method: [] for method in config["methods"]}
            for impact in config["impacts"]:
                low, high = [target_lookup[(members[member], impact)] for member in ("low", "high")]
                difference = actual["gaps"][low]-actual["gaps"][high]
                differences.append(float(np.sqrt(np.mean(difference**2))*1000))
                event_disagreements += int(actual["failure"][low] != actual["failure"][high])
                if equal:
                    squared_differences.append(float(np.mean(difference**2)/4))
                for method in config["methods"]:
                    rows = [prediction_lookup[(members[m], method, impact)] for m in ("low", "high")]
                    if all(row["representative_index"] is not None for row in rows):
                        ids = [row["candidate_rows"][row["representative_index"]] for row in rows]
                        representative_changes[method].append(float(np.abs(values["gaps"][ids[0]]-values["gaps"][ids[1]]).max()))
            pair_audit.append({"pair": pair, "probe": probe, "members": members, "observed_arrays_bitwise_identical": equal,
                "observed_rgb_max_difference": float(np.abs(observed[0]["rgb"]-observed[1]["rgb"]).max()),
                "truth_future_gap_rmse_between_members_mm": differences, "different_separation_outcomes": event_disagreements,
                "representative_prediction_max_difference_m": {m: max(v) if v else None for m, v in representative_changes.items()}})
        floors[str(probe)] = {"exactly_equal_observation_pairs": sum(r["probe"] == probe and r["observed_arrays_bitwise_identical"] for r in pair_audit),
            "balanced_equal_pair_point_rmse_floor_mm": float(np.sqrt(np.mean(squared_differences))*1000) if squared_differences else None,
            "scope": "Only exactly identical observed pairs, equally weighted true members and the configured three impacts. A statistical point-prediction lower bound for this constructed subset, not a universal physical error floor."}
    records = []
    for row in index:
        target = target_lookup[(row["case"], row["impact"])]
        accepted = row["accepted_indices"]
        if not accepted or row["representative_index"] is None:
            continue
        candidates = np.array(row["candidate_rows"])[accepted]
        representative = row["candidate_rows"][row["representative_index"]]
        curves = values["gaps"][candidates]
        lower, upper = curves.min(0), curves.max(0)
        definition = case_lookup[row["case"]]["methods"][row["method"]]
        parameters = np.array(definition["parameters"])
        estimate = parameters[row["representative_index"]]
        truth = labels[row["case"]]
        records.append({"case": row["case"], "method": row["method"], "probe": row["probe"], "impact": row["impact"],
            "representative_mse_m2": float(np.mean((values["gaps"][representative]-actual["gaps"][target])**2)),
            "representative_separation_correct": bool(values["failure"][representative] == actual["failure"][target]),
            "candidate_band_time_coverage": float(np.mean((actual["gaps"][target] >= lower)&(actual["gaps"][target] <= upper))),
            "candidate_band_width_mm": float((upper-lower).mean()*1000),
            "candidate_event_set_contains_truth": bool((values["failure"][candidates] == actual["failure"][target]).any()),
            "energy_abs_error": float(abs(estimate[1]-truth["energy"])), "history_abs_error_mm": float(abs(estimate[0]-truth["peak"])*1000),
            "accepted_count": len(accepted), "accepted_energy_span": float(np.ptp(parameters[accepted, 1]))})
    summaries = {}
    for probe in config["probes"]:
        common = {case["id"] for case in packet["cases"] if case["probe"] == probe and
            all(case["methods"][method]["representative_index"] is not None for method in config["methods"])}
        summaries[str(probe)] = {"common_cases": len(common), "methods": {}}
        for method in config["methods"]:
            selected = [r for r in records if r["probe"] == probe and r["method"] == method and r["case"] in common]
            summaries[str(probe)]["methods"][method] = {"forecasts": len(selected),
                "representative_gap_rmse_mm": float(np.sqrt(np.mean([r["representative_mse_m2"] for r in selected]))*1000),
                "representative_separation_accuracy": float(np.mean([r["representative_separation_correct"] for r in selected])),
                **{key: float(np.mean([r[key] for r in selected])) for key in ("candidate_band_time_coverage", "candidate_band_width_mm", "candidate_event_set_contains_truth", "energy_abs_error", "history_abs_error_mm", "accepted_count", "accepted_energy_span")}}
    # First pair, weak probe, first impact: fixed before outcome inspection.
    demo_pair = next(r for r in pair_audit if r["pair"] == 0 and r["probe"] == config["probes"][0])
    low_case, high_case = demo_pair["members"]["low"], demo_pair["members"]["high"]
    impact = config["impacts"][0]
    low, high = target_lookup[(low_case, impact)], target_lookup[(high_case, impact)]
    demo_prediction = prediction_lookup[(low_case, "directL49", impact)]
    representative = demo_prediction["candidate_rows"][demo_prediction["representative_index"]]
    accepted_rows = np.array(demo_prediction["candidate_rows"])[demo_prediction["accepted_indices"]]
    np.savez_compressed(root/"demo.npz", truth_low=actual["gaps"][low], truth_high=actual["gaps"][high],
        truth_low_xleft=actual["x_left"][low], truth_low_xright=actual["x_right"][low],
        truth_high_xleft=actual["x_left"][high], truth_high_xright=actual["x_right"][high],
        representative=values["gaps"][representative], representative_xleft=values["x_left"][representative], representative_xright=values["x_right"][representative],
        candidate_lower=values["gaps"][accepted_rows].min(0), candidate_upper=values["gaps"][accepted_rows].max(0))
    write_json(root/"demo.json", {"low_case": low_case, "high_case": high_case, "impact": impact,
        "rule": "First pair, weak probe, first impact, DIRECT-L representative; fixed before evaluation"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for j, method in enumerate(config["methods"]):
        axes[0].bar(np.arange(len(config["probes"]))+(j-1.5)*.18,
            [summaries[str(p)]["methods"][method]["representative_gap_rmse_mm"] for p in config["probes"]], .18, label=method)
    axes[0].set_xticks([0, 1], ["Weak observed probe", "Stronger observed probe"])
    axes[0].set_ylabel("Representative future gap RMSE (mm)")
    curves = np.load(root/"demo.npz")
    axes[1].fill_between(config["target_times"], curves["candidate_lower"]*1000, curves["candidate_upper"]*1000, alpha=.18, label="Accepted candidate range")
    for key in ("truth_low", "truth_high", "representative"):
        axes[1].plot(config["target_times"], curves[key]*1000, label=key)
    axes[1].set_xlabel("Elapsed time (s)"); axes[1].set_ylabel("Same observed prefix, future gap (mm)")
    for method in config["methods"]:
        record = case_lookup[low_case]["methods"][method]
        params = np.array(record["parameters"])
        axes[2].scatter(params[:, 0]*1000, params[:, 1], s=12, alpha=.7, label=method)
    for label in (labels[low_case], labels[high_case]):
        axes[2].scatter(label["peak"]*1000, label["energy"], marker="x", s=65, color="black")
    axes[2].set_xlabel("Reachable prehistory peak (mm)"); axes[2].set_ylabel("Candidate fracture energy (J/m2)")
    for axis in axes:
        axis.legend(fontsize=6); axis.grid(alpha=.2)
    fig.savefig(root/"joint-comparison.png", dpi=150); plt.close(fig)
    report = {"status": "joint_damage_material_diagnostic_completed", "run_directory": str(root), "summaries": summaries,
        "paired_observation_audit": pair_audit, "paired_point_prediction_floors": floors, "per_forecast_results": records,
        "probe_added_damage_history_counts": {str(p): sum(label["probe"] == p and label["probe_added_damage_history"] for label in labels.values()) for p in config["probes"]},
        "checks": {"true_energy_absent_from_public_model": "fracture_energy" not in packet["known_model"]["law"],
            "callback_budget_respected": all(r["budget_audit"]["callback_evaluations"] <= config["budget"] for c in packet["cases"] for r in c["methods"].values()),
            "inference_excludes_true_parameters_and_future_responses": not packet["private_parameters_or_responses_read"],
            "selection_precedes_future_plans": not packet["future_plans_read"], "material_and_state_carried_together": True},
        "inference": {k: packet[k] for k in ("wall_seconds", "cache_statistics", "pretrained_or_neural_model_used")},
        "unique_material_state_recovery_claimed": False, "novel_method_validated": False,
        "limitations": ["Matched-stiffness pairs are designed identifiability controls, not an independent generalization benchmark.",
            "No calibrated probabilities: accepted candidate bands/event sets are finite hypotheses only.",
            "The stronger probe is preset and can cause additional damage; no safe or learned active probing policy is claimed.",
            "Only two rigid blocks with one fixed interface; no general 3D fracture, real material calibration or learned world-model benefit.",
            "DIRECT and adaptive subdivision are conventional optimizers; novelty is not established."]}
    write_json(root/"report.json", report)
    out = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/joint-damage-material"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "joint-comparison.png", "config.json", "demo.json"):
        shutil.copy2(root/name, out/name)
    print("JOINT_EVALUATED "+json.dumps({"summaries": summaries, "floors": floors, "checks": report["checks"]}), flush=True)


def render(root):
    from cohesive_probe import BlockObservations
    from PIL import Image, ImageDraw
    packet = sealed(root)
    config = json.loads((root/"config.json").read_text())
    demo = json.loads((root/"demo.json").read_text())
    observer = BlockObservations(packet["known_model"])
    width = config["render_width"]
    with np.load(root/"demo.npz", allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    frames = root/"video-frames"
    frames.mkdir(exist_ok=True)
    panels = [("truth_low", "True lower fracture energy"), ("truth_high", "True higher fracture energy"), ("representative", "One DIRECT-L representative")]
    for t, elapsed in enumerate(config["target_times"]):
        canvas = Image.new("RGB", (3*width, width+64), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (key, title) in enumerate(panels):
            rgb = observer.capture({"x_left": values[key+"_xleft"][t], "x_right": values[key+"_xright"][t]})["rgb"]
            canvas.paste(Image.fromarray(np.round(np.clip(rgb, 0, 1)*255).astype(np.uint8)), (column*width, 0))
            draw.text((column*width+6, width+4), title, fill="black")
            draw.text((column*width+6, width+21), f"gap={values[key][t]*1000:.2f}mm", fill="black")
        draw.text((6, width+43), f"t={elapsed:.3f}s | matched observed prefix | new dv={demo['impact']:.3f}m/s", fill="black")
        canvas.save(frames/f"{t:05d}.png")
    canvas.save(root/"prediction-end.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i", str(frames/"%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(root/"matched-observation-futures.mp4")], check=True)
    out = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/joint-damage-material"
    for name in ("prediction-end.png", "matched-observation-futures.mp4"):
        shutil.copy2(root/name, out/name)
    print(f"GAUSSIAN_VIDEO={root/'matched-observation-futures.mp4'}", flush=True)
