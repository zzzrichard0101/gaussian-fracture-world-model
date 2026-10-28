"""Descriptive replication summary of already sealed refinement experiments."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from history_refinement import write_json


def summarize(runs):
    reports = [json.loads((root/"report.json").read_text()) for root in runs]
    methods = reports[0]["experiment"]["methods"]
    if any(report["experiment"]["search"] != reports[0]["experiment"]["search"] for report in reports):
        raise ValueError("Replication changed search hyperparameters")
    folds, pooled = [], {}
    for root, report in zip(runs, reports):
        folds.append({"run_directory": str(root), "seed": report["experiment"]["seed"],
            "informative_test_histories": report["common_informative_test_histories"],
            "methods": {method: report["summary"][method]["all"] for method in methods},
            "inference": report["inference"], "checks": report["checks"]})
    for method in methods:
        rows = [fold["methods"][method] for fold in folds]
        count = sum(row["forecast_cases"] for row in rows)
        pooled[method] = {"forecast_cases": count,
            "gap_rmse_mm": float(np.sqrt(sum(row["forecast_cases"]*row["gap_rmse_mm"]**2 for row in rows)/count)),
            "separation_accuracy": float(sum(row["forecast_cases"]*row["separation_accuracy"] for row in rows)/count)}
    output = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/history-refinement"
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "fixed_settings_replication_complete", "folds": folds, "pooled_descriptive_metrics": pooled,
        "search_settings_unchanged": True, "independent_test_histories": sum(f["informative_test_histories"] for f in folds),
        "new_method_novelty_validated": False, "pretrained_or_neural_model_used": False,
        "notes": ["Each histogram reuses three impacts per physical history. Pooling is descriptive, not a confidence interval.",
            "Equal budgets count distinct observation-loss evaluations, not identical actual runtime under the shared forward cache.",
            "Replication changes histories, observation-noise seed and sample count only; search rules were frozen.",
            "This is a known one-dimensional reduced fixture; real materials and general 3D fracture are unvalidated."]}
    write_json(output/"comparison-summary.json", result)
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x = np.arange(len(folds))
    labels = {"coarse13": "Fixed 13 (smaller budget)", "uniform49": "Uniform 49", "greedy49": "Single-basin 49", "multiregion49": "Multi-region 49"}
    for j, method in enumerate(methods):
        axis.bar(x+(j-1.5)*.19, [fold["methods"][method]["gap_rmse_mm"] for fold in folds], .19, label=labels[method])
    axis.set_xticks(x, [f"Primary: {f['informative_test_histories']} histories" if i == 0 else f"Replication: {f['informative_test_histories']} histories" for i, f in enumerate(folds)])
    axis.set_ylabel("Heldout future gap RMSE (mm)")
    axis.set_title("Observation-only refinement; three new impacts per history")
    axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)
    figure.savefig(output/"replication-comparison.png", dpi=160); plt.close(figure)
    print(json.dumps({"folds": [{"seed": f["seed"], "rmse_mm": {method: f["methods"][method]["gap_rmse_mm"] for method in methods}} for f in folds], "pooled": pooled}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    summarize(parser.parse_args().runs)
