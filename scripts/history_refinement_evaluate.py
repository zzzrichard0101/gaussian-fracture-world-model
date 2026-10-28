"""Compare frozen observation searches on the same private heldout responses."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from history_refinement import digest, sealed, write_json


def checked_forecasts(root):
    selection = sealed(root)
    path = root/"forecast/seal.json"
    if digest(path) != (root/"forecast/seal.sha256").read_text():
        raise ValueError("Forecast seal changed")
    seal = json.loads(path.read_text())
    for key, relative in [("selection_sha256", "fitted/selection.json"), ("future_plan_sha256", "future-plans.json"),
                          ("forecast_sha256", "forecast/point-forecasts.npz"), ("index_sha256", "forecast/index.json")]:
        if digest(root/relative) != seal[key]:
            raise ValueError("Sealed forecast dependency changed")
    with np.load(root/"forecast/point-forecasts.npz", allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    return selection, json.loads((root/"forecast/index.json").read_text()), values


def evaluate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected, predictions, values = checked_forecasts(root)
    experiment = json.loads((root/"experiment.json").read_text())
    methods = experiment["methods"]
    # Private histories/responses are first opened after selection and forecasting.
    with np.load(root/"private/test-targets.npz", allow_pickle=False) as archive:
        truth, failure = archive["gaps"], archive["failure"]
    history = json.loads((root/"private/history-labels.json").read_text())
    prefix_meta = json.loads((root/"prefix-index.json").read_text())
    target_index = json.loads((root/"inputs/test-index.json").read_text())
    target_lookup = {(f"prefix-{row['prefix_index']:03d}", row["impact"]): i for i, row in enumerate(target_index)}
    cases = {row["id"]: row for row in selected["cases"]}
    common = {row["id"] for row in selected["cases"] if row["split"] == "test" and row["probe"] > 0
        and all(row["methods"][method]["point_index"] is not None for method in methods)}
    records = []
    for index, row in enumerate(predictions):
        target = target_lookup[(row["case"], row["impact"])]
        prefix_number = int(row["case"].split("-")[-1])
        true_peak = history[prefix_meta[prefix_number]["history_group"]]
        records.append({**row, "common_case": row["case"] in common,
            "gap_mse_m2": float(np.mean((values["gaps"][index]-truth[target])**2)),
            "gap_rmse_mm": float(np.sqrt(np.mean((values["gaps"][index]-truth[target])**2))*1000),
            "history_abs_error_m": float(abs(row["point_peak"]-true_peak)),
            "separation_correct": bool(values["failure"][index] == failure[target])})
    summary = {}
    for method in methods:
        summary[method] = {}
        for name, impact in [("all", None)]+[(f"impact-{a:g}", a) for a in experiment["new_impacts"]]:
            rows = [r for r in records if r["method"] == method and r["common_case"] and (impact is None or r["impact"] == impact)]
            summary[method][name] = {"forecast_cases": len(rows),
                "gap_rmse_mm": float(np.sqrt(np.mean([r["gap_mse_m2"] for r in rows]))*1000) if rows else None,
                "median_case_gap_rmse_mm": float(np.median([r["gap_rmse_mm"] for r in rows])) if rows else None,
                "maximum_case_gap_rmse_mm": float(max(r["gap_rmse_mm"] for r in rows)) if rows else None,
                "history_mae_mm": float(np.mean([r["history_abs_error_m"] for r in rows])*1000) if rows else None,
                "separation_accuracy": float(np.mean([r["separation_correct"] for r in rows])) if rows else None}
    diagnostics = {}
    for method in methods:
        rows = [row["methods"][method] for row in selected["cases"]]
        diagnostics[method] = {"logical_evaluations_per_case": sorted(set(r["logical_history_evaluations"] for r in rows)),
            "logical_observation_renders_per_case": sorted(set(r["logical_observation_renders"] for r in rows)),
            "all_case_status_counts": {status: sum(r["status"] == status for r in rows) for status in ("matched", "ambiguous", "no_observation_match")},
            "stationary_unique_point_predictions": sum(row["probe"] == 0 and row["methods"][method]["point_index"] is not None for row in selected["cases"]),
            "mean_final_observation_score_probed": float(np.mean([row["methods"][method]["best_observed_score"] for row in selected["cases"] if row["probe"] > 0]))}
    # Fixed showcase, chosen without evaluating accuracy.
    demo = next(row for row in selected["cases"] if row["split"] == "test" and row["probe"] > 0)
    impact = experiment["new_impacts"][1]
    target = target_lookup[(demo["id"], impact)]
    demo_curves = {"truth": truth[target]}
    for method in methods:
        row = next(i for i, r in enumerate(predictions) if r["case"] == demo["id"] and r["method"] == method and r["impact"] == impact)
        demo_curves[method] = values["gaps"][row]
    np.savez_compressed(root/"demo-curves.npz", **demo_curves)
    write_json(root/"demo.json", {"case": demo["id"], "impact": impact, "probe": demo["probe"],
        "rule": "First probed test history, middle predeclared impact; not selected by accuracy"})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    x = np.arange(len(experiment["new_impacts"]))
    for j, method in enumerate(methods):
        axes[0].bar(x+(j-1.5)*.18, [summary[method][f"impact-{a:g}"]["gap_rmse_mm"] for a in experiment["new_impacts"]], .18, label=method)
    axes[0].set_xticks(x, [str(a) for a in experiment["new_impacts"]])
    axes[0].set_xlabel("Known new velocity increment (m/s)"); axes[0].set_ylabel("Future gap RMSE (mm)")
    plan = json.loads((root/"future-plans.json").read_text())
    for label, gap in demo_curves.items():
        axes[1].plot(plan["target_times"], gap*1000, label=label, linestyle="--" if label == "truth" else "-")
    axes[1].set_xlabel("Elapsed time (s)"); axes[1].set_ylabel("Showcase interface gap (mm)")
    for method in methods:
        record = demo["methods"][method]
        axes[2].scatter(np.array(record["peaks"])*1000, [r["score"] for r in record["metrics"]], s=12, alpha=.7, label=method)
    axes[2].set_xlabel("Evaluated reachable history peak (mm)"); axes[2].set_ylabel("Prefix RGB-D residual score")
    for axis in axes:
        axis.grid(alpha=.2); axis.legend(fontsize=7)
    figure.savefig(root/"refinement-comparison.png", dpi=150); plt.close(figure)
    report = {"status": "observation_only_history_refinement_executed", "run_directory": str(root), "experiment": experiment,
        "summary": summary, "diagnostics": diagnostics, "common_informative_test_histories": len(common),
        "inference": {key: selected[key] for key in ("cache_statistics", "wall_seconds", "peak_torch_allocated_mib", "future_plan_read", "private_labels_read", "pretrained_or_neural_module_used")},
        "data": json.loads((root/"data-summary.json").read_text()), "per_case_results": records,
        "checks": {"all_49_methods_respect_equal_logical_budget": all(diagnostics[m]["logical_evaluations_per_case"] == [49] for m in methods if m != "coarse13"),
            "stationary_observations_not_given_unique_states": all(d["stationary_unique_point_predictions"] == 0 for d in diagnostics.values()),
            "selection_sealed_before_future_plan": not selected["future_plan_read"], "private_labels_excluded_from_inference": not selected["private_labels_read"]},
        "novel_method_validated": False, "general_3d_fracture_validated": False,
        "limitations": ["Known material, camera, geometry, initial pose and observed impulse; only one unknown prehistory peak.",
            "Methods have equal logical prefix-loss evaluation budgets; cache reuse and batching make actual runtime unequal.",
            "This stage runs physics/Gaussian inference only, without V-JEPA or learned heads.",
            "Adaptive one-dimensional observation-loss search is a conventional baseline, not a novelty claim.",
            "Stationary cases retain multiple cutoff states and receive no unique point forecast; primary accuracy uses common matched informative test cases.",
            "Alternative states are retained, but their finite envelope is not a calibrated uncertainty interval.",
            "One fresh synthetic split and three impacts; no real-material or arbitrary 3D fracture validation."]}
    write_json(root/"report.json", report)
    target_directory = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/history-refinement"
    target_directory.mkdir(parents=True, exist_ok=True)
    run_directory = target_directory/root.name
    run_directory.mkdir(exist_ok=True)
    for name in ("report.json", "refinement-comparison.png", "experiment.json", "demo.json"):
        shutil.copy2(root/name, target_directory/name)
        shutil.copy2(root/name, run_directory/name)
    write_json(target_directory/"latest.json", {"run_directory": str(root), "report": str(run_directory/"report.json")})
    print("REFINEMENT_EVALUATED "+json.dumps({"checks": report["checks"], "summary": summary}), flush=True)


def render(root):
    from cohesive_probe import BlockObservations
    from PIL import Image, ImageDraw
    selected, predictions, values = checked_forecasts(root)
    demo = json.loads((root/"demo.json").read_text())
    experiment = json.loads((root/"experiment.json").read_text())
    plan = json.loads((root/"future-plans.json").read_text())
    with np.load(root/"demo-curves.npz", allow_pickle=False) as archive:
        truth = archive["truth"]
    observer = BlockObservations(selected["model"])
    width = selected["model"]["render"]["width"]
    method_rows = {method: next(i for i, r in enumerate(predictions) if r["case"] == demo["case"] and r["method"] == method and r["impact"] == demo["impact"]) for method in experiment["methods"]}
    frames = root/"video-frames"
    frames.mkdir(exist_ok=True)
    labels = ["Synthetic reference", *experiment["methods"]]
    mass = selected["model"]["mass_left"]+selected["model"]["mass_right"]
    model_config = json.loads((root/"config.json").read_text())
    for t, elapsed in enumerate(plan["target_times"]):
        center = selected["model"]["mass_left"]/mass*(demo["probe"]*max(0, elapsed-model_config["probe_time"])+demo["impact"]*max(0, elapsed-plan["time"]))
        gap = truth[t]
        reference = {"x_left": center-selected["model"]["mass_right"]/mass*(selected["model"]["width"]+gap),
                     "x_right": center+selected["model"]["mass_left"]/mass*(selected["model"]["width"]+gap)}
        canvas = Image.new("RGB", (width*len(labels), width+64), "white")
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(labels):
            if column == 0:
                pose, gap = reference, truth[t]
            else:
                row = method_rows[label]
                pose = {"x_left": values["x_left"][row, t], "x_right": values["x_right"][row, t]}
                gap = values["gaps"][row, t]
            rgb = observer.capture(pose)["rgb"]
            canvas.paste(Image.fromarray(np.round(np.clip(rgb, 0, 1)*255).astype(np.uint8)), (column*width, 0))
            draw.text((column*width+6, width+4), label, fill="black")
            draw.text((column*width+6, width+21), f"gap={gap*1000:.3f} mm", fill="black")
        draw.text((6, width+43), f"t={elapsed:.3f}s | 1D fixture, slow playback | impact dv={demo['impact']:.3f}m/s", fill="black")
        canvas.save(frames/f"{t:05d}.png")
    canvas.save(root/"prediction-end.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i", str(frames/"%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(root/"prediction-comparison.mp4")], check=True)
    output = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/history-refinement"
    run_output = output/root.name
    run_output.mkdir(exist_ok=True)
    for name in ("prediction-end.png", "prediction-comparison.mp4"):
        shutil.copy2(root/name, output/name)
        shutil.copy2(root/name, run_output/name)
    print(f"GAUSSIAN_VIDEO={root/'prediction-comparison.mp4'}", flush=True)
