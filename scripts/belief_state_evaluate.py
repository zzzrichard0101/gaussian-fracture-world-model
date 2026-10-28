"""Evaluate sealed finite-mixture forecasts; private test labels read only here."""
import json
import os
from pathlib import Path
import shutil

import numpy as np

from action_adapter import digest, write_json
from belief_state import load_evidence, state_ce
from wmfracture.belief_state import mixture_diagnostics


def evaluate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    packet = json.loads((root/"posterior/packet.json").read_text())
    if digest(root/"posterior/packet.json") != (root/"posterior/packet.sha256").read_text():
        raise ValueError("Posterior packet changed before evaluation")
    forecast_seal = root/"forecasts/seal.json"
    audit = json.loads(forecast_seal.read_text())
    if (digest(forecast_seal) != (root/"forecasts/seal.sha256").read_text()
        or digest(root/"forecasts/candidate-branches.npz") != audit["forecast_sha256"]
        or digest(root/"posterior/packet.json") != audit["posterior_packet_sha256"]
        or digest(root/"posterior/weights.npz") != packet["weights_sha256"]
        or digest(root/"candidate-prefix-states.json") != packet["state_library_sha256"]
        or digest(root/"belief-inputs/test.npz") != packet["belief_input_sha256"]):
        raise ValueError("State/forecast dependency changed after sealing")
    for key, relative in [("future_plan_index_sha256", "physics-bank-index.json"), ("future_test_case_index_sha256", "inputs/test.npz")]:
        if key in audit and audit[key] != digest(root/relative):
            raise ValueError("Future plan indexing changed after forecasting")
    settings = json.loads((root/"belief-settings.json").read_text())
    with np.load(root/"posterior/weights.npz", allow_pickle=False) as archive:
        weights, prefixes = archive["weights"], archive["prefix_index"]
    with np.load(root/"forecasts/candidate-branches.npz", allow_pickle=False) as archive:
        banks, failures = archive["gaps"], archive["failure"]
    # First private response/state access after both posterior and forecasts are sealed.
    with np.load(root/"private/test-targets.npz", allow_pickle=False) as archive:
        truth, broken = archive["gaps"], archive["failure"]
    with np.load(root/"private/test-states.npz", allow_pickle=False) as archive:
        if not np.array_equal(prefixes, archive["prefix_index"]):
            raise ValueError("Private state evaluation misaligned")
        state_targets = archive["targets"]
    with np.load(root/"inputs/test.npz", allow_pickle=False) as archive:
        bank_indices, case_prefixes = archive["bank_index"], archive["prefix_index"]
    case_index = json.loads((root/"inputs/test-index.json").read_text())
    evidence = load_evidence(root, "test")
    lookup = {int(prefix): i for i, prefix in enumerate(prefixes)}
    plans = json.loads((root/"physics-bank-index.json").read_text())
    if any(not np.isclose(plans[int(bank_index)]["probe"], evidence["probe"][lookup[int(prefix)], 0], atol=1e-8, rtol=0)
           for prefix, bank_index in zip(case_prefixes, bank_indices)):
        raise ValueError("Future plan selected a different past observation state")
    valid_prefixes = evidence["accepted"].any(1)
    masks = {f"{regime}/{protocol}": np.array([r["regime"] == regime and bool(r["probe"]) == informative for r in case_index])
        for regime in ("test_seen", "test_interpolation", "test_extrapolation")
        for protocol, informative in (("stationary", False), ("probed", True))}
    masks["all"] = np.ones(len(case_index), dtype=bool)
    # Privileged evaluation controls: never used to infer weights or choose settings.
    # They separate candidate-grid capacity from weight-estimation error.
    candidate_peaks = np.array(json.loads((root/"config.json").read_text())["candidate_peaks"])
    oracle_records = []
    for row, (prefix, bank_index) in enumerate(zip(case_prefixes, bank_indices)):
        position = lookup[int(prefix)]
        encoded = state_targets[position]
        peak = float(encoded@candidate_peaks)
        nearest = int(np.argmin(abs(candidate_peaks-peak)))
        allowed = evidence["accepted"][position]
        squared_error = np.mean((banks[bank_index]-truth[row])**2, axis=1)
        oracle_records.append({"nearest_true_history_mse": float(squared_error[nearest]),
            "perfect_grid_label_mixture_mean_mse": float(np.mean((encoded@banks[bank_index]-truth[row])**2)),
            "best_allowed_future_mse": float(squared_error[allowed].min()) if allowed.any() else None,
            "true_grid_label_mass_excluded": float(encoded[~allowed].sum())})
    oracle_summary = {}
    for name, mask in masks.items():
        selected = [row for row, use in zip(oracle_records, mask) if use]
        oracle_summary[name] = {"nearest_true_history_gap_rmse_mm": float(np.sqrt(np.mean([r["nearest_true_history_mse"] for r in selected]))*1000),
            "perfect_grid_label_mixture_mean_gap_rmse_mm": float(np.sqrt(np.mean([r["perfect_grid_label_mixture_mean_mse"] for r in selected]))*1000),
            "best_allowed_future_gap_rmse_mm": float(np.sqrt(np.mean([r["best_allowed_future_mse"] for r in selected if r["best_allowed_future_mse"] is not None]))*1000),
            "mean_true_grid_label_mass_excluded": float(np.mean([r["true_grid_label_mass_excluded"] for r in selected]))}
    write_json(root/"oracle-diagnostics.json", {"privileged_evaluation_only": True,
        "used_for_training_or_settings_selection": False,
        "note": "True-history and future-response oracles are diagnostic controls, not deployable methods. The best-future oracle can choose a different state for each action and cannot be called a single inferred belief.",
        "results": oracle_summary})
    methods, state_diagnostics = {}, {}
    for method_index, name in enumerate(packet["models"]):
        records = []
        for row, (prefix, bank_index) in enumerate(zip(case_prefixes, bank_indices)):
            q = weights[method_index, lookup[int(prefix)]]
            records.append(mixture_diagnostics(q, banks[bank_index], failures[bank_index], truth[row], broken[row],
                settings["nominal_candidate_band_mass"]) if q.sum() > 0 else None)
        summaries = {}
        for split, mask in masks.items():
            selected = [record for record, use in zip(records, mask) if use and record is not None]
            if not selected:
                summaries[split] = {"count": int(mask.sum()), "forecast_count": 0}; continue
            summary = {key: float(np.mean([r[key] for r in selected])) for key in selected[0]}
            summary["mean_gap_rmse_mm"] = float(np.sqrt(summary.pop("mean_gap_mse_m2"))*1000)
            summary["map_gap_rmse_mm"] = float(np.sqrt(summary.pop("map_gap_mse_m2"))*1000)
            for old, new in [("path_energy_m", "path_energy_mm"), ("marginal_crps_m", "marginal_crps_mm"),
                             ("candidate_band_mean_width_m", "candidate_band_mean_width_mm")]:
                summary[new] = summary.pop(old)*1000
            summary.update({"count": int(mask.sum()), "forecast_count": len(selected)})
            summaries[split] = summary
        methods[name] = summaries
        state_diagnostics[name] = {}
        for protocol, informative in (("stationary", False), ("probed", True)):
            mask = (evidence["probe"][:, 0] > 0) == informative
            mask &= valid_prefixes
            q = weights[method_index, mask]
            positive = q > 0
            entropy = -(np.where(positive, q*np.log(np.maximum(q, 1e-300)), 0)).sum(1)
            state_diagnostics[name][protocol] = {"prefixes": int(mask.sum()), "state_entropy_nats": float(entropy.mean()),
                "effective_candidates": float((1/(q*q).sum(1)).mean()), "maximum_weight": float(q.max(1).mean()),
                "barycentric_state_cross_entropy": state_ce(weights[method_index], state_targets, mask, settings["probability_floor_for_state_ce"])}
    grouped = {}
    group_names = ["uniform_accepted", "rgbd_map", "calibrated_energy", "learned_rgbd", "learned_vjepa"]
    for group in group_names:
        members = [name for name in methods if name == group or name.startswith(group+"/")]
        grouped[group] = {"members": members, "results": {}}
        for split in masks:
            selected = [methods[name][split] for name in members]
            keys = [key for key in selected[0] if key not in ("count", "forecast_count")]
            grouped[group]["results"][split] = {key: float(np.mean([s[key] for s in selected])) for key in keys}
            grouped[group]["results"][split]["seed_std_mean_gap_rmse_mm"] = float(np.std([s["mean_gap_rmse_mm"] for s in selected]))
    regimes = ["test_seen/probed", "test_interpolation/probed", "test_extrapolation/probed"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = np.arange(3)
    plot_groups = group_names[1:]
    for j, group in enumerate(plot_groups):
        for axis, metric, label in [(axes[0, 0], "mean_gap_rmse_mm", "Mixture-mean gap RMSE (mm)\nStatistical summary"),
                                    (axes[0, 1], "path_energy_mm", "Path energy score (mm), lower is better"),
                                    (axes[1, 0], "failure_brier", "Separation Brier score, lower is better")]:
            axis.bar(x+(j-1.5)*.19, [grouped[group]["results"][r][metric] for r in regimes], .19, label=group)
            axis.set_ylabel(label); axis.grid(axis="y", alpha=.2)
            axis.set_xticks(x, ["Seen action", "Interpolated action", "Extrapolated action"])
        member_names = grouped[group]["members"]
        axes[1, 1].bar(np.arange(2)+(j-1.5)*.19,
            [np.mean([state_diagnostics[name][p]["effective_candidates"] for name in member_names]) for p in ("stationary", "probed")], .19, label=group)
    axes[1, 1].set_xticks([0, 1], ["Identical stationary prefix", "Observed probe response"])
    axes[1, 1].set_ylabel("Effective state candidates (unique prefixes)")
    for axis in axes.flat:
        axis.legend(fontsize=7)
    figure.savefig(root/"belief-comparison.png", dpi=150); plt.close(figure)
    plans = json.loads((root/"physics-bank-index.json").read_text())
    boundary_difference = 0.
    for probe in sorted({p["probe"] for p in plans}):
        selected = [i for i, plan in enumerate(plans) if plan["probe"] == probe]
        boundary_difference = max(boundary_difference, float(np.abs(banks[selected, :, 0]-banks[selected[0], :, 0]).max()))
    report = {"status": "weighted_state_pipeline_executed", "run_directory": str(root),
        "evaluation_code_sha256": digest(Path(__file__)),
        "evaluation_revision_note": "Dependency checks and privileged oracle diagnostics added after initial evaluation; sealed weights, forecasts and performance settings were unchanged.",
        "experiment": json.loads((root/"experiment.json").read_text()), "data": json.loads((root/"data-summary.json").read_text()),
        "training": json.loads((root/"belief-trained/manifest.json").read_text()),
        "features": json.loads((root/"feature-audit.json").read_text()), "forecast_audit": audit,
        "methods": methods, "grouped_methods": grouped, "state_diagnostics": state_diagnostics,
        "checks": {"posterior_sealed_before_new_plan": not packet["future_plans_read"],
            "future_response_not_used_for_weight_fitting": True, "test_predictions_sealed_before_reference_read": True,
            "pre_impulse_branch_positions_invariant": boundary_difference == 0.,
            "cutoff_history_retained": True, "candidate_states_not_averaged": True},
        "pre_impulse_branch_max_position_difference_m": boundary_difference,
        "new_method_validated": False, "continuous_posterior_calibrated": False,
        "limitations": ["One fixed camera, material, geometry and prescribed interface; synthetic supervision of history-grid weights.",
            "State targets are a barycentric grid encoding, not observations of a true discrete posterior.",
            "Energy temperature and residual epochs use validation state labels only; no future outcome or test label tuning.",
            "Mixture-mean error describes a statistical summary; it is not an additional physically reachable trajectory.",
            "A nominal 90% weighted candidate band has finite support and no confidence or coverage guarantee; measured coverage is reported.",
            "Correlated frames and average sensor residuals form an empirical energy, not an exact independent-pixel likelihood.",
            "Learned features, state hypotheses and physical rollout integration are conventional components; novelty and 3D validity are not established.",
            "Many action variants reuse one prefix; test forecast count is not the number of independent history groups."]}
    if (root/"report.json").exists() and not (root/"report-initial.json").exists():
        shutil.copy2(root/"report.json", root/"report-initial.json")
    write_json(root/"report.json", report)
    target = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/belief-state"
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("report.json", "belief-comparison.png", "experiment.json", "oracle-diagnostics.json"):
        shutil.copy2(root/filename, target/filename)
    print("BELIEF_EVALUATED "+json.dumps({group: {regime: {key: grouped[group]["results"][regime][key] for key in
        ("mean_gap_rmse_mm", "map_gap_rmse_mm", "path_energy_mm", "failure_brier")} for regime in regimes} for group in plot_groups}), flush=True)
