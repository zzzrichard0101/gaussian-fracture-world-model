"""Actual pretrained V-JEPA inference + observation-constrained physics ranking.

No RGB decoder or numeric-force action interface is claimed for base V-JEPA 2.
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
import urllib.request

import h5py
import numpy as np

from wmfracture.world_model_io import hybrid_choice, validate_rank_arrays


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def config_default():
    return json.loads((Path(os.environ["WM_PROJECT_ROOT"]) / "config/world-model.json").read_text())


def weights():
    config = config_default()
    path = Path(os.environ["WM_RESEARCH_ROOT"]) / config["weights_relative_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".pt.partial")
        urllib.request.urlretrieve(config["weights_url"], temporary)
        if temporary.stat().st_size != config["weights_bytes"] or digest(temporary) != config["weights_sha256"]:
            raise ValueError("Downloaded checkpoint does not match pinned official bytes")
        temporary.rename(path)
    if digest(path) != config["weights_sha256"]:
        raise ValueError("Existing official checkpoint checksum mismatch")
    print(f"WEIGHTS_VERIFIED={path}")


def rank(root):
    import torch
    from wmfracture.vjepa_world import VJEPAWorld
    config = json.loads((root / "config.json").read_text())
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.mem_get_info()[0]
    start = time.perf_counter()
    model = VJEPAWorld(config)
    records = []
    output = root / "ranked"
    output.mkdir(exist_ok=False)
    for identifier in json.loads((root / "dataset-index.json").read_text()):
        folder = root / "inputs" / identifier
        meta = json.loads((folder / "manifest.json").read_text())
        if digest(folder / "rank-input.npz") != meta["rank_arrays_sha256"]:
            raise ValueError("Prefix/candidate inputs changed after export")
        with np.load(folder / "rank-input.npz", allow_pickle=False) as archive:
            arrays = validate_rank_arrays({key: archive[key] for key in archive.files}, config)
        predicted = model.predict(arrays["context_rgb"])
        changed_filler = model.predict(arrays["context_rgb"], filler=1.)
        invariance = float((predicted - changed_filler).abs().max())
        if invariance > config["filler_invariance_tolerance"]:
            raise ValueError("Unknown future filler changed pretrained prediction")
        persistence = model.persistence(arrays["context_rgb"])
        roi = model.roi_weights(arrays["context_roi"])
        energies = [model.energy(predicted, model.encode_targets(clip), roi) for clip in arrays["candidate_rgb"]]
        latent = int(np.argmin([row["roi_mae"] for row in energies]))
        hybrid = hybrid_choice(meta["rgbd_accepted_indices"], [row["roi_mae"] for row in energies])
        np.savez_compressed(output / f"{identifier}-prediction.npz", predicted=predicted.cpu().numpy(),
                            persistence=persistence.cpu().numpy(), roi=roi.cpu().numpy())
        records.append({"sample": identifier, "rgbd_best_index": meta["rgbd_best_index"],
            "rgbd_accepted_indices": meta["rgbd_accepted_indices"], "learned_prior_index": latent, "hybrid_index": hybrid,
            "candidate_energies": energies, "prediction_shape": list(predicted.shape),
            "unknown_filler_max_absolute_change": invariance,
            "context_times": meta["context_times"], "latent_horizon_times": meta["latent_horizon_times"],
            "real_suffix_read": False, "private_physics_read": False})
        print(f"RANKED {identifier} RGBD={meta['rgbd_best_index']} prior={latent} hybrid={hybrid} filler_delta={invariance:g}", flush=True)
    torch.cuda.synchronize()
    report = {"schema": "pretrained-vjepa-ranking-1", "config": config, "pretrained_load_audit": model.load_audit,
        "samples": records, "wall_seconds_including_model_load": time.perf_counter() - start,
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "free_device_memory_before_mib": before / 2**20,
        "note": "Prediction and all selections made before reading heldout real suffix/physical reference."}
    write_json(output / "selection.json", report)
    (output / "selection.sha256").write_text(digest(output / "selection.json"))


def rows(path):
    with h5py.File(path) as stream:
        return {key: stream[key][:] for key in stream}


def evaluate(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from wmfracture.vjepa_world import VJEPAWorld
    path = root / "ranked/selection.json"
    if digest(path) != (root / "ranked/selection.sha256").read_text().strip():
        raise ValueError("Selection changed before evaluation")
    selection = json.loads(path.read_text())
    config = selection["config"]
    model = VJEPAWorld(config)
    results = []
    for record in selection["samples"]:
        identifier = record["sample"]
        with np.load(root / "evaluation" / identifier / "real-observed-clip.npz", allow_pickle=False) as archive:
            reference_clip = archive["rgb"]
        actual_tokens = model.encode_targets(reference_clip)
        with np.load(root / "ranked" / f"{identifier}-prediction.npz", allow_pickle=False) as archive:
            prediction = torch.tensor(archive["predicted"], device="cuda")
            persistence = torch.tensor(archive["persistence"], device="cuda")
            roi = torch.tensor(archive["roi"], device="cuda")
        predicted_error = model.energy(prediction, actual_tokens, roi)
        persistence_error = model.energy(persistence, actual_tokens, roi)
        # Stronger persistence control in the same full-clip target representation.
        # Use only the observed prefix, never copy the withheld real suffix.
        with np.load(root / "inputs" / identifier / "rank-input.npz", allow_pickle=False) as archive:
            prefix = archive["context_rgb"]
        frozen_clip = np.concatenate([prefix, np.repeat(prefix[-1:], config["observation_slots"]-len(prefix), axis=0)])
        frozen_rgb_error = model.energy(model.encode_targets(frozen_clip), actual_tokens, roi)
        meta = json.loads((root / "inputs" / identifier / "manifest.json").read_text())
        protocol = json.loads((root / "evaluation" / identifier / "metadata.json").read_text())["protocol"]
        truth = rows(root / "evaluation" / identifier / "truth.h5")
        future_start = min(action["time"] for action in meta["future_input"]["actions"])
        future_mask = truth["elapsed"] > future_start
        entries = {}
        for name, index in [("rgbd_prefix", record["rgbd_best_index"]), ("learned_prior_ranking", record["learned_prior_index"]), ("hybrid", record["hybrid_index"])]:
            if index is None:
                entries[name] = None
                continue
            trajectory = rows(root / "physics-candidates" / identifier / f"candidate-{index:02d}.h5")
            if not np.allclose(trajectory["elapsed"], truth["elapsed"], atol=1e-12, rtol=0):
                raise ValueError("Physical prediction/reference clocks differ")
            entries[name] = {"candidate_index": index, "history_peak": meta["candidate_peaks"][index],
                "future_gap_rmse_m": float(np.sqrt(np.mean((trajectory["opening"][future_mask] - truth["opening"][future_mask])**2))),
                "fully_separated": bool(trajectory["damage"][-1] == 1.)}
        results.append({"sample": identifier, "observation_protocol": protocol, "latent_prediction_mae": predicted_error,
                        "latent_persistence_mae": persistence_error, "frozen_rgb_suffix_latent_mae": frozen_rgb_error,
                        "physical_future_results": entries})
        print(f"EVALUATED {identifier} latent ROI prediction={predicted_error['roi_mae']:.4f} feature_persistence={persistence_error['roi_mae']:.4f} frozen_rgb={frozen_rgb_error['roi_mae']:.4f}", flush=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(results))
    axes[0].bar(x-.24, [row["latent_prediction_mae"]["roi_mae"] for row in results], .24, label="Pretrained predictor")
    axes[0].bar(x, [row["latent_persistence_mae"]["roi_mae"] for row in results], .24, label="Context feature persistence")
    axes[0].bar(x+.24, [row["frozen_rgb_suffix_latent_mae"]["roi_mae"] for row in results], .24, label="Frozen last RGB frame")
    axes[0].set_ylabel("Heldout suffix latent MAE (observed ROI)")
    for offset, name, label in [(-.24, "rgbd_prefix", "RGB-D prefix ranking"), (0, "learned_prior_ranking", "Learned prior + physics bank"), (.24, "hybrid", "RGB-D constraints + learned prior")]:
        axes[1].bar(x+offset, [row["physical_future_results"][name]["future_gap_rmse_m"]*1000 if row["physical_future_results"][name] else np.nan for row in results], .24, label=label)
    axes[1].set_ylabel("New-impulse gap RMSE (mm), physics rollouts")
    for axis in axes:
        axis.set_xticks(x, [row["sample"] for row in results]); axis.legend(fontsize=7); axis.grid(axis="y", alpha=.2)
    figure.savefig(root / "pretrained-comparison.png", dpi=150)
    plt.close(figure)
    checks = {"official_pretrained_encoder_and_predictor_loaded": all(not value["missing_keys"] for value in selection["pretrained_load_audit"].values()),
        "strict_future_filler_invariance": all(row["unknown_filler_max_absolute_change"] <= config["filler_invariance_tolerance"] for row in selection["samples"]),
        "ranking_sealed_before_reference_read": True, "all_finite": all(np.isfinite(row["latent_prediction_mae"]["roi_mae"]) for row in results)}
    report = {"status": "pretrained_integration_checks_passed" if all(checks.values()) else "pretrained_integration_checks_failed",
        "run_directory": str(root), "checks": checks, "model": config, "runtime": selection,
        "results": results, "new_method_validated": False, "force_conditioned_world_model_alone_comparison_completed": False,
        "evaluation_code_sha256": digest(Path(__file__)),
        "evaluation_revision_note": "Frozen-last-RGB control added after initial evaluation to check whether the feature-persistence baseline was too weak. Sealed candidate selections were not changed.",
        "limitations": ["Frozen masked-latent prediction protocol; not a native numeric-force-conditioned world model.",
            "No pixel decoder. Physics candidates and Gaussian renderer supply physical trajectories/images.",
            "All three physical-future ranking variants share the same physics bank and same four-frame context.",
            "Model-alone metric is latent prediction vs persistence on a no-new-action suffix, not standalone impulse-to-video prediction.",
            "Target embeddings attend to the complete clip; repeating masked-context embeddings is a weaker control than encoding a prefix plus frozen-RGB suffix in that same representation.",
            "Only four synthetic reduced-interface sequences. Natural-video pretraining, 16-frame repeated input and short physical time scales create domain shift.",
            "Integration, loading and no-future-input checks are not proof of improved forecasting or novelty."]}
    write_json(root / "report.json", report)
    target = Path(os.environ["WM_PROJECT_ROOT"]) / "artifacts/world-model"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "pretrained-comparison.png", "config.json"):
        shutil.copy2(root / name, target / name)
    print(json.dumps({"status": report["status"], "checks": checks, "results": results}, indent=2), flush=True)


def launch(command, log):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in process.stdout:
            stream.write(line); tail = (tail+[line])[-20:]
            if line.startswith(("EXPORTED ", "RANKED ", "EVALUATED ")):
                print(line, end="", flush=True)
        if process.wait():
            print("".join(tail), flush=True)
            raise subprocess.CalledProcessError(process.returncode, command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["weights", "rank", "evaluate"])
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.stage == "weights": weights(); return
    if args.stage == "rank": rank(args.run); return
    if args.stage == "evaluate" or args.run: evaluate(args.run); return
    project = Path(os.environ["WM_PROJECT_ROOT"])
    root = Path(os.environ["WM_RESEARCH_ROOT"]) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-world-model")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config_default())
    write_json(root / "local-code-sha256.json", {str(path.relative_to(project)): digest(path)
        for path in [Path(__file__), project / "scripts/world_model_export.py", project / "wmfracture/vjepa_world.py", project / "wmfracture/world_model_io.py"]})
    print(f"OUTPUT={root}", flush=True)
    # Separate CUDA environments communicate only through numeric array files.
    launch(["bash", str(project / "scripts/wsl.sh"), "python", str(project / "scripts/world_model_export.py"), "--run", str(root)], root / "export.log")
    for stage in ("rank", "evaluate"):
        launch([sys.executable, "-u", str(Path(__file__)), "--stage", stage, "--run", str(root)], root / f"{stage}.log")


if __name__ == "__main__":
    main()
