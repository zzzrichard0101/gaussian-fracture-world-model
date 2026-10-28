"""Frozen V-JEPA -> small action dynamics -> observation-constrained physics.

Supervised reduced-scene baseline and ablations, not a claimed novel method.
Stages isolate training, prediction sealing and private test evaluation.
"""
import argparse
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

from wmfracture.action_data import assert_group_split, project_to_bank, validate_inputs


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_inputs(root, split):
    with np.load(root/"inputs"/f"{split}.npz", allow_pickle=False) as archive:
        return validate_inputs({key: archive[key] for key in archive.files})


def features(root):
    import torch
    from wmfracture.vjepa_world import VJEPAWorld
    start = time.perf_counter()
    config = json.loads((root/"pretrained-config.json").read_text())
    model = VJEPAWorld(config)
    torch.cuda.reset_peak_memory_stats()
    index = json.loads((root/"prefix-index.json").read_text())
    embeddings = []
    # Predict every prefix independently; neither training labels nor heldout suffix are read.
    for record in index:
        with np.load(root/"prefixes"/f"prefix-{record['index']:03d}.npz", allow_pickle=False) as archive:
            rgb = archive["rgb"]
            roi = model.roi_weights(archive["alpha"].max(0) > .5)
        prediction = model.predict(rgb)
        embeddings.append(((prediction[0]*roi[:, None]).sum(0)/roi.sum()).cpu().numpy())
        if len(embeddings) % 8 == 0:
            print(f"FEATURES {len(embeddings)}/{len(index)}", flush=True)
    np.save(root/"frozen-predictor-features.npy", np.stack(embeddings).astype(np.float32), allow_pickle=False)
    write_json(root/"feature-audit.json", {"official_model_load": model.load_audit,
        "input": "observed four-frame RGB prefix and its foreground mask only",
        "predictor_executed": True, "backbone_training": False, "feature_shape": list(np.stack(embeddings).shape),
        "wall_seconds": time.perf_counter()-start, "peak_torch_allocated_mib": torch.cuda.max_memory_allocated()/2**20})


def train(root):
    import torch
    import torch.nn.functional as F
    from wmfracture.action_adapter import DynamicsHead, fit_pca, model_inputs, training_normalizer
    start = time.perf_counter()
    config = json.loads((root/"config.json").read_text())
    if config["target_times"][0] != config["future_time"] or any(t <= config["future_time"] for t in config["target_times"][1:]):
        raise ValueError("This fixed-clock causal head requires its first position at the pre-impulse boundary")
    assert_group_split(json.loads((root/"split-groups.json").read_text()))
    # These are the ONLY labeled files opened by this stage.
    inputs = {split: load_inputs(root, split) for split in ("train", "validation")}
    labels = {}
    for split in inputs:
        with np.load(root/"supervision"/f"{split}-targets.npz", allow_pickle=False) as archive:
            labels[split] = {key: torch.tensor(archive[key], device="cuda") for key in archive.files}
    embeddings = np.load(root/"frozen-predictor-features.npy", allow_pickle=False)
    center, components = fit_pca(embeddings, inputs["train"]["prefix_index"], config["pca_dimensions"])
    output = root/"trained"
    output.mkdir()
    np.savez_compressed(output/"preprocessing.npz", pca_center=center, pca_components=components)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    records = []
    for kind in config["models"]:
        raw = {split: model_inputs(kind, values, embeddings, center, components) for split, values in inputs.items()}
        mean, std = training_normalizer(raw["train"])
        x = {split: torch.tensor((value-mean)/std, device="cuda") for split, value in raw.items()}
        for seed in config["training_seeds"]:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            future_column = None if kind.endswith("no_future_action") else x["train"].shape[1]-1
            model = DynamicsHead(x["train"].shape[1], len(config["target_times"]), config["hidden_width"], future_column).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            best_score, best_state, best_epoch = float("inf"), None, None
            trace = []
            for epoch in range(config["epochs"]):
                model.train()
                order = torch.randperm(len(x["train"]), device="cuda")
                for indices in order.split(config["batch_size"]):
                    predicted = model(x["train"][indices])
                    gap_loss = F.mse_loss(predicted[:, :-1], labels["train"]["gaps"][indices]/config["gap_scale_m"])
                    fail_loss = F.binary_cross_entropy_with_logits(predicted[:, -1], labels["train"]["failure"][indices])
                    loss = gap_loss+config["failure_loss_weight"]*fail_loss
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                if epoch % 5 == 0 or epoch == config["epochs"]-1:
                    model.eval()
                    with torch.no_grad():
                        score = float(F.mse_loss(model(x["validation"])[:, :-1], labels["validation"]["gaps"]/config["gap_scale_m"]))
                    if not np.isfinite(score):
                        raise ValueError("Nonfinite validation loss")
                    trace.append({"epoch": epoch+1, "validation_scaled_gap_mse": score})
                    if score < best_score:
                        best_score, best_epoch = score, epoch+1
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            relative = f"{kind}-seed-{seed}.pt"
            torch.save({"state_dict": best_state, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                "input_size": raw["train"].shape[1], "kind": kind, "seed": seed,
                "future_action_column": future_column, "causal_boundary_version": 2}, output/relative)
            records.append({"model": kind, "seed": seed, "checkpoint": relative, "sha256": digest(output/relative),
                "parameters": sum(value.numel() for value in model.parameters()), "selected_epoch": best_epoch,
                "best_validation_scaled_gap_mse": best_score, "trace": trace})
            print(f"TRAINED {kind} seed={seed} epoch={best_epoch} val={best_score:.5g}", flush=True)
    write_json(output/"manifest.json", {"records": records, "pca_fit_on": "unique training prefixes only",
        "normalization_fit_on": "training rows only", "test_labels_read": False,
        "validation_labels_used_for_epoch_selection": True, "epochs": config["epochs"],
        "wall_seconds": time.perf_counter()-start, "torch": torch.__version__, "device": torch.cuda.get_device_name(),
        "peak_torch_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
        "preprocessing_sha256": digest(output/"preprocessing.npz")})


def predict(root):
    import torch
    from wmfracture.action_adapter import DynamicsHead, model_inputs
    config = json.loads((root/"config.json").read_text())
    manifest = json.loads((root/"trained/manifest.json").read_text())
    inputs = load_inputs(root, "test")
    embeddings = np.load(root/"frozen-predictor-features.npy", allow_pickle=False)
    with np.load(root/"trained/preprocessing.npz", allow_pickle=False) as archive:
        center, components = archive["pca_center"], archive["pca_components"]
    with np.load(root/"physics-bank.npz", allow_pickle=False) as archive:
        bank = archive["gaps"][inputs["bank_index"]]
    output = root/"predictions"
    output.mkdir()
    audit = []
    for record in manifest["records"]:
        path = root/"trained"/record["checkpoint"]
        if digest(path) != record["sha256"] or digest(root/"trained/preprocessing.npz") != manifest["preprocessing_sha256"]:
            raise ValueError("Trained model/preprocessing changed before test prediction")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        values = model_inputs(record["model"], inputs, embeddings, center, components)
        values = (values-checkpoint["mean"].numpy())/checkpoint["std"].numpy()
        if checkpoint.get("causal_boundary_version") != 2:
            raise ValueError("Noncausal v1 heads require their preserved predictions, not this v2 prediction path")
        model = DynamicsHead(checkpoint["input_size"], len(config["target_times"]), config["hidden_width"],
                             checkpoint["future_action_column"]).cuda().eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        with torch.inference_mode():
            predictions = model(torch.tensor(values, device="cuda"))
            gap = predictions[:, :-1].cpu().numpy()*config["gap_scale_m"]
            fail = predictions[:, -1].sigmoid().cpu().numpy()
        if not np.isfinite(gap).all() or not np.isfinite(fail).all():
            raise ValueError("Nonfinite test predictions")
        indices = np.array([project_to_bank(values, accepted, candidates)
            for values, accepted, candidates in zip(gap, inputs["accepted"], bank)])
        projected = np.stack([candidates[index] if index >= 0 else np.full(candidates.shape[1], np.nan)
                             for candidates, index in zip(bank, indices)])
        filename = f"{record['model']}-seed-{record['seed']}.npz"
        np.savez_compressed(output/filename, gaps=gap, failure_probability=fail, projected_indices=indices, projected_gaps=projected)
        audit.append({"model": record["model"], "seed": record["seed"], "file": filename, "sha256": digest(output/filename)})
    seal = {"predictions": audit, "test_labels_read": False, "training_manifest_sha256": digest(root/"trained/manifest.json"),
        "test_inputs_sha256": digest(root/"inputs/test.npz"), "features_sha256": digest(root/"frozen-predictor-features.npy"),
        "physics_bank_sha256": digest(root/"physics-bank.npz")}
    write_json(output/"seal.json", seal)
    (output/"seal.sha256").write_text(digest(output/"seal.json"))
    print(f"PREDICTIONS_SEALED models={len(audit)} test_cases={len(inputs['actions'])}", flush=True)


def evaluate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    config = json.loads((root/"config.json").read_text())
    if digest(root/"predictions/seal.json") != (root/"predictions/seal.sha256").read_text():
        raise ValueError("Prediction seal changed")
    seal = json.loads((root/"predictions/seal.json").read_text())
    for key, relative in [("training_manifest_sha256", "trained/manifest.json"), ("test_inputs_sha256", "inputs/test.npz"),
                          ("features_sha256", "frozen-predictor-features.npy"), ("physics_bank_sha256", "physics-bank.npz")]:
        if digest(root/relative) != seal[key]:
            raise ValueError(f"Changed sealed dependency: {relative}")
    # First access to test response labels happens after all model predictions are sealed.
    with np.load(root/"private/test-targets.npz", allow_pickle=False) as archive:
        truth = {key: archive[key] for key in archive.files}
    inputs = load_inputs(root, "test")
    index = json.loads((root/"inputs/test-index.json").read_text())
    with np.load(root/"physics-bank.npz", allow_pickle=False) as archive:
        banks = archive["gaps"][inputs["bank_index"]]
    baseline = np.stack([bank[best] if accepted.any() else np.full(bank.shape[1], np.nan)
                         for bank, best, accepted in zip(banks, inputs["rgbd_best"], inputs["accepted"])])
    predictions = []
    for item in seal["predictions"]:
        if digest(root/"predictions"/item["file"]) != item["sha256"]:
            raise ValueError("Test predictions modified")
        with np.load(root/"predictions"/item["file"], allow_pickle=False) as archive:
            predictions.append((item, {key: archive[key] for key in archive.files}))
    slices = {f"{regime}/{protocol}": np.array([row["regime"] == regime and bool(row["probe"]) == informative for row in index])
        for regime in ("test_seen", "test_interpolation", "test_extrapolation")
        for protocol, informative in (("stationary", False), ("probed", True))}
    slices["all"] = np.ones(len(index), dtype=bool)
    rmse = lambda values, mask: float(np.sqrt(np.mean((values[mask]-truth["gaps"][mask])**2))*1000)
    results = {}
    for name, mask in slices.items():
        valid = mask & np.isfinite(baseline).all(1)
        records = {}
        for kind in config["models"]:
            runs = []
            for item, values in predictions:
                if item["model"] != kind:
                    continue
                projected_valid = mask & (values["projected_indices"] >= 0)
                runs.append({"seed": item["seed"], "raw_gap_rmse_mm": rmse(values["gaps"], mask),
                    "projected_gap_rmse_mm": rmse(values["projected_gaps"], projected_valid) if projected_valid.any() else None,
                    "failure_accuracy": float(np.mean((values["failure_probability"][mask] >= .5) == truth["failure"][mask])),
                    "failure_brier": float(np.mean((values["failure_probability"][mask]-truth["failure"][mask])**2)),
                    "projected_count": int(projected_valid.sum())})
            records[kind] = {"runs": runs,
                "raw_gap_rmse_mm_mean": float(np.mean([row["raw_gap_rmse_mm"] for row in runs])),
                "raw_gap_rmse_mm_seed_std": float(np.std([row["raw_gap_rmse_mm"] for row in runs])),
                "projected_gap_rmse_mm_mean": float(np.mean([row["projected_gap_rmse_mm"] for row in runs])) if all(row["projected_gap_rmse_mm"] is not None for row in runs) else None}
        results[name] = {"count": int(mask.sum()), "fully_separated_reference_count": int(truth["failure"][mask].sum()),
            "rgbd_physics_gap_rmse_mm": rmse(baseline, valid) if valid.any() else None,
            "physics_match_count": int(valid.sum()), "models": records}
    action_sensitivity, boundary_sensitivity = {}, {}
    for item, values in predictions:
        differences, boundary_changes = [], []
        for prefix in np.unique(inputs["prefix_index"]):
            pair = [next(i for i, row in enumerate(index) if row["prefix_index"] == prefix and row["impact"] == action) for action in (.12, .36)]
            differences.append(float(np.sqrt(np.mean((values["gaps"][pair[1]]-values["gaps"][pair[0]])**2))*1000))
            boundary_changes.append(float(abs(values["gaps"][pair[1], 0]-values["gaps"][pair[0], 0])*1000))
        action_sensitivity[f"{item['model']}/seed-{item['seed']}"] = float(np.mean(differences))
        boundary_sensitivity[f"{item['model']}/seed-{item['seed']}"] = float(max(boundary_changes))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    regimes = ["test_seen/probed", "test_interpolation/probed", "test_extrapolation/probed"]
    colors = ["#999999", "#387ab8", "#df893a", "#65a163"]
    x = np.arange(len(regimes))
    for j, (kind, color) in enumerate(zip(config["models"], colors)):
        axes[0].bar(x+(j-1.5)*.18, [results[r]["models"][kind]["raw_gap_rmse_mm_mean"] for r in regimes], .18,
            yerr=[results[r]["models"][kind]["raw_gap_rmse_mm_seed_std"] for r in regimes], color=color, label=kind, capsize=2)
    axes[1].bar(x-.24, [results[r]["rgbd_physics_gap_rmse_mm"] for r in regimes], .24, label="RGB-D physics baseline")
    for offset, kind in [(0, "rgbd_action"), (.24, "rgbd_vjepa_action")]:
        axes[1].bar(x+offset, [results[r]["models"][kind]["projected_gap_rmse_mm_mean"] if results[r]["models"][kind]["projected_gap_rmse_mm_mean"] is not None else np.nan for r in regimes], .24, label=kind+" + physics projection")
    for ax in axes:
        ax.set_xticks(x, ["Heldout history\nseen action", "Heldout history\ninterpolated action", "Heldout history\nextrapolated action"])
        ax.set_ylabel("Future gap RMSE (mm)"); ax.grid(axis="y", alpha=.2); ax.legend(fontsize=6)
    axes[0].set_title("Direct learned dynamics (probed test cases)")
    axes[1].set_title("Same reachable physics bank (probed test cases)")
    figure.savefig(root/"action-comparison.png", dpi=160); plt.close(figure)
    # Predeclared showcase: first probed test history, seen action 0.24, first seed.
    demo = next(i for i, row in enumerate(index) if row["probe"] == .04 and row["impact"] == .24)
    demo_item, demo_values = next((item, values) for item, values in predictions
        if item["model"] == "rgbd_vjepa_action" and item["seed"] == config["training_seeds"][0])
    np.savez_compressed(root/"demo-trajectories.npz", truth=truth["gaps"][demo], baseline=baseline[demo],
        raw=demo_values["gaps"][demo], projected=demo_values["projected_gaps"][demo])
    write_json(root/"demo.json", index[demo])
    report = {"status": "action_adapter_probe_completed", "run_directory": str(root), "config": config,
        "data": json.loads((root/"data-summary.json").read_text()),
        "features": json.loads((root/"feature-audit.json").read_text()),
        "training": json.loads((root/"trained/manifest.json").read_text()), "results": results,
        "paired_action_change_rmse_mm": action_sensitivity,
        "paired_pre_impulse_position_change_max_mm": boundary_sensitivity,
        "checks": {"history_groups_disjoint": True, "test_labels_read_during_training_or_prediction": False,
            "predictions_sealed_before_test_evaluation": True, "pretrained_backbone_frozen": True,
            "no_future_action_head_is_invariant": all(value == 0 for name, value in action_sensitivity.items() if "no_future_action" in name),
            "pre_impulse_position_is_causal": all(value == 0 for value in boundary_sensitivity.values())},
        "new_method_validated": False,
        "limitations": ["One known material, camera, geometry and prescribed 1D interface; 44 synthetic history groups only.",
            "Small supervised fixed-horizon heads, not an independently pretrained general world model.",
            "Numeric actions are prescribed velocity increments, not simulated impactor contact or language actions.",
            "All baseline and projection methods share the physical bank; projection adds no simulator speed benefit here.",
            "PCA/normalization use training inputs, epochs use validation responses. No hyperparameter tuning on test results.",
            "Stationary prefixes are identical across histories and cannot uniquely identify damage.",
            "Raw learned gaps/failure scores do not guarantee contact feasibility, damage irreversibility or calibrated probabilities.",
            "Three training seeds quantify optimization variation, not a statistical confidence interval or broad generalization.",
            "Conventional frozen-feature heads, supervised dynamics and candidate projection are not claimed as novel."]}
    write_json(root/"report.json", report)
    target = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/action-adapter"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "action-comparison.png", "config.json"):
        shutil.copy2(root/name, target/name)
    print(json.dumps({"status": report["status"], "checks": report["checks"], "results": results}, indent=2), flush=True)


def launch(command, log):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in process.stdout:
            stream.write(line); stream.flush(); tail = (tail+[line])[-20:]
            if line.startswith(("PREFIXES", "DATA_READY", "FEATURES", "TRAINED", "PREDICTIONS_SEALED", "GAUSSIAN_VIDEO", "BELIEF", "FORECAST", "BRANCH")):
                print(line, end="", flush=True)
        if process.wait():
            print("".join(tail), flush=True)
            raise subprocess.CalledProcessError(process.returncode, command)


def render(root):
    project = Path(os.environ["WM_PROJECT_ROOT"])
    launch(["bash", str(project/"scripts/wsl.sh"), "python", str(project/"scripts/action_adapter_render.py"),
            "--run", str(root)], root/"render.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["features", "train", "predict", "evaluate", "render", "continue"])
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    project = Path(os.environ["WM_PROJECT_ROOT"])
    if args.stage and args.stage != "continue":
        globals()[args.stage](args.run); return
    if args.run:
        root = args.run
        # Attach the pinned pretrained configuration to a generated dataset.
        if not (root/"pretrained-config.json").exists():
            shutil.copy2(project/"config/world-model.json", root/"pretrained-config.json")
    else:
        root = Path(os.environ["WM_RESEARCH_ROOT"])/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-action-adapter")
        root.mkdir(parents=True)
        shutil.copy2(project/"config/action-adapter.json", root/"config.json")
        shutil.copy2(project/"config/world-model.json", root/"pretrained-config.json")
        print(f"OUTPUT={root}", flush=True)
        launch(["bash", str(project/"scripts/wsl.sh"), "python", str(project/"scripts/action_adapter_data.py"), "--run", str(root)], root/"generate.log")
    write_json(root/"code-sha256.json", {str(path.relative_to(project)): digest(path) for path in
        [Path(__file__), project/"scripts/action_adapter_data.py", project/"wmfracture/action_adapter.py",
         project/"wmfracture/action_data.py", project/"wmfracture/cohesive_batch.py", project/"wmfracture/vjepa_world.py",
         project/"scripts/action_adapter_render.py"]})
    for stage in ("features", "train", "predict", "evaluate"):
        launch([sys.executable, "-u", str(Path(__file__)), "--stage", stage, "--run", str(root)], root/f"{stage}.log")
    render(root)


if __name__ == "__main__":
    main()
