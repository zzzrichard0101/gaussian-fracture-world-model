"""Frozen DIRECT baselines on the earlier one-dimensional public observations."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from history_refinement import ObservationEngine, digest, write_json, launch, forecast, sealed
from wmfracture.budgeted_direct import run_direct
from wmfracture.history_refinement import canonical_peak, selection


def infer(root):
    import scipy
    header = json.loads((root/"inference-inputs.json").read_text())
    settings = json.loads((root/"search-settings.json").read_text())
    config = json.loads((root/"optimizer-settings.json").read_text())
    engine = ObservationEngine(root, header, settings, 32)
    start = time.perf_counter()
    cases, timing = [], {}
    for case in header["cases"]:
        records = {}
        for method in config["methods"]:
            began = time.perf_counter()
            def objective(x):
                peak = canonical_peak(x[0])
                engine.fulfil([(case["id"], peak)])
                return engine.book[(case["id"], peak)]["score"]
            run = run_direct(objective, [(settings["lower"], settings["upper"])], config["budget"],
                method == "directL49", config["optimizer"])
            peaks = sorted({canonical_peak(row[0]) for row in run["samples"]})
            metrics = [engine.book[(case["id"], peak)] for peak in peaks]
            records[method] = {**selection(metrics, settings), "peaks": peaks, "metrics": metrics,
                "cutoff_states": [engine.predictions[(case["probe"], peak)]["state"] for peak in peaks],
                "budget_audit": run, "wall_seconds_with_shared_cache": time.perf_counter()-began}
            timing[method] = timing.get(method, 0.)+records[method]["wall_seconds_with_shared_cache"]
        cases.append({"id": case["id"], "split": "test", "probe": case["probe"], "methods": records})
        print(f"DIRECT_CASE {len(cases)}/{len(header['cases'])}", flush=True)
    output = root/"fitted"
    output.mkdir()
    packet = {"schema": "direct-observation-budget-comparison-1", "cases": cases, "model": header["model"],
        "observation_end": max(header["times"]), "scipy": scipy.__version__, "wall_seconds": time.perf_counter()-start,
        "method_elapsed_with_shared_cache": timing, "cache_statistics": engine.stats,
        "input_sha256": digest(root/"inference-inputs.json"), "optimizer_settings_sha256": digest(root/"optimizer-settings.json"),
        "observation_hashes": engine.input_hashes, "future_plan_read": False, "private_labels_read": False}
    write_json(output/"selection.json", packet)
    (output/"selection.sha256").write_text(digest(output/"selection.json"))
    print("DIRECT_SEALED", flush=True)


def evaluate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    packet = sealed(root)
    audit = json.loads((root/"forecast/seal.json").read_text())
    for key, relative in [("selection_sha256", "fitted/selection.json"), ("future_plan_sha256", "future-plans.json"),
                          ("forecast_sha256", "forecast/point-forecasts.npz"), ("index_sha256", "forecast/index.json")]:
        if digest(root/relative) != audit[key]:
            raise ValueError("DIRECT test predictions changed before evaluation")
    index = json.loads((root/"forecast/index.json").read_text())
    with np.load(root/"forecast/point-forecasts.npz", allow_pickle=False) as archive:
        gaps, failure = archive["gaps"], archive["failure"]
    sources = json.loads((root/"source-map.json").read_text())
    folds, records = [], []
    for fold_index, source in enumerate(sources):
        source = Path(source)
        source_report = json.loads((source/"report.json").read_text())
        with np.load(source/"private/test-targets.npz", allow_pickle=False) as archive:
            actual, broken = archive["gaps"], archive["failure"]
        target_index = json.loads((source/"inputs/test-index.json").read_text())
        lookup = {(f"prefix-{row['prefix_index']:03d}", row["impact"]): i for i, row in enumerate(target_index)}
        prefix = f"fold-{fold_index}-"
        for i, row in enumerate(index):
            if not row["case"].startswith(prefix):
                continue
            case = row["case"][len(prefix):]
            target = lookup[(case, row["impact"])]
            records.append({**row, "fold": fold_index, "source_case": case,
                "gap_mse_m2": float(np.mean((gaps[i]-actual[target])**2)),
                "separation_correct": bool(failure[i] == broken[target])})
        common = {case["id"][len(prefix):] for case in packet["cases"] if case["id"].startswith(prefix)
            and all(value["point_index"] is not None for value in case["methods"].values())}
        methods = {}
        for method in source_report["experiment"]["methods"]:
            rows = [r for r in source_report["per_case_results"] if r["method"] == method and r["case"] in common]
            methods[method] = {"forecast_cases": len(rows), "gap_rmse_mm": float(np.sqrt(np.mean([r["gap_mse_m2"] for r in rows]))*1000),
                "separation_accuracy": float(np.mean([r["separation_correct"] for r in rows]))}
        for method in ("direct49", "directL49"):
            rows = [r for r in records if r["fold"] == fold_index and r["method"] == method and r["source_case"] in common]
            methods[method] = {"forecast_cases": len(rows), "gap_rmse_mm": float(np.sqrt(np.mean([r["gap_mse_m2"] for r in rows]))*1000),
                "separation_accuracy": float(np.mean([r["separation_correct"] for r in rows]))}
        folds.append({"source_run": str(source), "common_test_histories": len(common), "methods": methods})
    methods = list(folds[0]["methods"])
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for j, method in enumerate(methods):
        axis.bar(np.arange(len(folds))+(j-(len(methods)-1)/2)*.13,
            [fold["methods"][method]["gap_rmse_mm"] for fold in folds], .13, label=method)
    axis.set_xticks([0, 1], ["Earlier primary observations", "Earlier replication observations"])
    axis.set_ylabel("Common-case future gap RMSE (mm)"); axis.legend(fontsize=7); axis.grid(axis="y", alpha=.2)
    figure.savefig(root/"direct-comparison.png", dpi=160); plt.close(figure)
    checks = {"callback_budget_respected": all(r["budget_audit"]["callback_evaluations"] <= 49 for c in packet["cases"] for r in c["methods"].values()),
        "selection_precedes_future_plan": not packet["future_plan_read"], "labels_excluded_from_search": not packet["private_labels_read"]}
    report = {"run_directory": str(root), "status": "direct_baseline_comparison_completed", "folds": folds,
        "checks": checks, "inference": {k: packet[k] for k in ("scipy", "wall_seconds", "method_elapsed_with_shared_cache", "cache_statistics")},
        "evaluation_counts": {m: sorted({c["methods"][m]["budget_audit"]["callback_evaluations"] for c in packet["cases"]}) for m in ("direct49", "directL49")},
        "per_case_results": records, "novelty_validated": False,
        "limitations": ["Earlier test observations are reused to extend the benchmark, not claimed as a new holdout.",
            "All methods have a 49-callback cap except coarse13. Duplicate/cached DIRECT calls are charged. Shared-cache timing is order-dependent.",
            "Callback truncation preserves best actually evaluated candidate, not scipy result.x or a convergence guarantee.",
            "Known 1D synthetic material/geometry only; no neural model or 3D fracture validation."]}
    write_json(root/"report.json", report)
    out = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/direct-benchmark"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "direct-comparison.png", "config.json"):
        shutil.copy2(root/name, out/name)
    print("DIRECT_EVALUATED "+json.dumps({"folds": folds, "checks": checks}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--stage", choices=["infer", "forecast", "evaluate"])
    parser.add_argument("--sources", nargs=2, type=Path, help="Two locally regenerated history-refinement run folders")
    args = parser.parse_args()
    if args.stage:
        {"infer": infer, "forecast": forecast, "evaluate": evaluate}[args.stage](args.run); return
    project = Path(os.environ["WM_PROJECT_ROOT"])
    config = json.loads((project/"config/direct-benchmark.json").read_text())
    root = Path(os.environ["WM_RESEARCH_ROOT"])/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-direct-benchmark")
    root.mkdir(parents=True)
    write_json(root/"config.json", config)
    write_json(root/"optimizer-settings.json", {key: config[key] for key in ("methods", "budget", "optimizer")})
    sources = args.sources or [root.parent/name for name in config["source_runs"]]
    if not all((source/"observation-manifest.json").is_file() for source in sources):
        parser.error("Generate the two history-refinement fixtures first and provide their run folders with --sources RUN1 RUN2")
    write_json(root/"source-map.json", [str(path) for path in sources])
    header, settings = None, None
    for fold, source in enumerate(sources):
        observed = json.loads((source/"observation-manifest.json").read_text())
        if header is None:
            header = {key: value for key, value in observed.items() if key != "cases"}; header["cases"] = []
            settings = json.loads((source/"search-settings.json").read_text())
        elif observed["model"] != header["model"] or observed["times"] != header["times"]:
            raise ValueError("Source fixtures differ")
        header["cases"].extend([{**case, "id": f"fold-{fold}-"+case["id"], "rgbd_file": str(source/case["rgbd_file"])}
                               for case in observed["cases"] if case["split"] == "test" and case["probe"] > 0])
    write_json(root/"inference-inputs.json", header)
    write_json(root/"search-settings.json", settings)
    shutil.copy2(sources[0]/"future-plans.json", root/"future-plans.json")
    write_json(root/"code-sha256.json", {str(p.relative_to(project)): digest(p) for p in [Path(__file__), project/"wmfracture/budgeted_direct.py", project/"scripts/history_refinement.py"]})
    print(f"OUTPUT={root}", flush=True)
    for stage in ("infer", "forecast", "evaluate"):
        launch([sys.executable, "-u", str(Path(__file__)), "--stage", stage, "--run", str(root)], root/f"{stage}.log")


if __name__ == "__main__":
    main()
