"""Observation-only state weights -> sealed reachable states -> physics branches."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from action_adapter import digest, write_json, launch, features
from wmfracture.action_data import assert_group_split
from wmfracture.belief_state import energy_weights, validate_evidence


SETTING_KEYS = ("version", "pca_dimensions", "training_seeds", "models", "hidden_width", "epochs", "learning_rate",
    "weight_decay", "residual_logit_bound", "prior_pseudocount_per_candidate", "temperature_grid", "probability_floor_for_state_ce", "nominal_candidate_band_mass")


def load_evidence(root, split):
    with np.load(root/"belief-inputs"/f"{split}.npz", allow_pickle=False) as archive:
        return validate_evidence({key: archive[key] for key in archive.files})


def vector(kind, evidence, embeddings, center, components):
    blocks = [evidence["rgbd_features"], evidence["probe"]]
    if kind == "learned_vjepa":
        blocks.append((embeddings[evidence["prefix_index"]]-center)@components.T)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def gate(scores, scale):
    spread = np.ptp(scores, axis=1)
    return (spread/(spread+scale)).astype(np.float32)


def state_ce(weights, targets, valid, floor):
    return float(-(targets[valid]*np.log(np.maximum(weights[valid], floor))).sum(1).mean())


def train(root):
    import torch
    from wmfracture.action_adapter import fit_pca, training_normalizer
    from wmfracture.belief_network import BeliefCorrection
    start = time.perf_counter()
    settings = json.loads((root/"belief-settings.json").read_text())
    if set(settings) != set(SETTING_KEYS):
        raise ValueError("Belief settings must exclude future plans")
    assert_group_split(json.loads((root/"split-groups.json").read_text()))
    evidence = {split: load_evidence(root, split) for split in ("train", "validation")}
    targets = {}
    for split, values in evidence.items():
        with np.load(root/"state-supervision"/f"{split}-states.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["prefix_index"], values["prefix_index"]):
                raise ValueError("State supervision/prefix alignment mismatch")
            targets[split] = archive["targets"]
    index = json.loads((root/"prefix-index.json").read_text())
    group_ids = [index[int(i)]["history_group"] for i in evidence["train"]["prefix_index"]]
    # Each history contributes once to the empirical prior despite paired probe observations.
    firsts = [group_ids.index(group) for group in sorted(set(group_ids))]
    counts = targets["train"][firsts].sum(0)+settings["prior_pseudocount_per_candidate"]
    prior = counts/counts.sum()
    spread = np.ptp(evidence["train"]["scores"], axis=1)
    scale = float(np.median(spread[spread > 0]))
    valid = {split: values["accepted"].any(1) for split, values in evidence.items()}
    grid = settings["temperature_grid"]
    trials = []
    for temperature in np.logspace(grid["log10_min"], grid["log10_max"], grid["count"]):
        weights = energy_weights(evidence["validation"]["scores"], evidence["validation"]["accepted"], prior, temperature)
        trials.append({"temperature": float(temperature), "validation_state_ce": state_ce(weights, targets["validation"], valid["validation"], settings["probability_floor_for_state_ce"])})
    selected = min(trials, key=lambda item: item["validation_state_ce"])
    temperature = selected["temperature"]
    embeddings = np.load(root/"frozen-predictor-features.npy", allow_pickle=False)
    center, components = fit_pca(embeddings, evidence["train"]["prefix_index"], settings["pca_dimensions"])
    output = root/"belief-trained"
    output.mkdir()
    np.savez_compressed(output/"preprocessing.npz", pca_center=center, pca_components=components, prior=prior,
        temperature=temperature, information_scale=scale)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    def tensor(value):
        return torch.as_tensor(value, device="cuda")
    base, masks, target_tensors, gates = {}, {}, {}, {}
    for split, values in evidence.items():
        base[split] = tensor((np.log(prior)[None]-(values["scores"]-values["scores"].min(1, keepdims=True))/temperature).astype(np.float32))
        masks[split] = tensor(values["accepted"].astype(bool))
        target_tensors[split] = tensor(targets[split].astype(np.float32))
        gates[split] = tensor(gate(values["scores"], scale))
    records = []
    for kind in settings["models"]:
        raw = {split: vector(kind, values, embeddings, center, components) for split, values in evidence.items()}
        mean, std = training_normalizer(raw["train"])
        x = {split: tensor((values-mean)/std) for split, values in raw.items()}
        def loss(model, split):
            logits = base[split]+model(x[split], gates[split])
            logits = logits.masked_fill(~masks[split], -1e30)
            weights = logits.softmax(-1).clamp_min(settings["probability_floor_for_state_ce"])
            rows = -(target_tensors[split]*weights.log()).sum(-1)
            return rows[tensor(valid[split])].mean()
        for seed in settings["training_seeds"]:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = BeliefCorrection(raw["train"].shape[1], settings["hidden_width"], settings["residual_logit_bound"]).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
            with torch.no_grad():
                best_score = float(loss(model, "validation"))
            best_epoch = 0
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            trace = [{"epoch": 0, "validation_state_ce": best_score}]
            for epoch in range(settings["epochs"]):
                model.train(); optimizer.zero_grad(set_to_none=True)
                value = loss(model, "train"); value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                if (epoch+1) % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        score = float(loss(model, "validation"))
                    if not np.isfinite(score):
                        raise ValueError("Nonfinite belief validation loss")
                    trace.append({"epoch": epoch+1, "validation_state_ce": score})
                    if score < best_score:
                        best_score, best_epoch = score, epoch+1
                        best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            filename = f"{kind}-seed-{seed}.pt"
            torch.save({"state_dict": best, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                "input_size": raw["train"].shape[1], "kind": kind}, output/filename)
            records.append({"model": kind, "seed": seed, "file": filename, "sha256": digest(output/filename),
                "selected_epoch": best_epoch, "validation_state_ce": best_score, "trace": trace,
                "parameters": sum(parameter.numel() for parameter in model.parameters())})
            print(f"BELIEF_TRAINED {kind} seed={seed} epoch={best_epoch} val={best_score:.5f}", flush=True)
    write_json(output/"manifest.json", {"models": records, "temperature_selection": trials,
        "temperature": temperature, "empirical_prior_history_groups": len(firsts),
        "preprocessing_sha256": digest(output/"preprocessing.npz"),
        "state_supervision_files": {split: digest(root/"state-supervision"/f"{split}-states.npz") for split in evidence},
        "future_response_labels_read": False, "future_plans_read": False, "test_labels_read": False,
        "wall_seconds": time.perf_counter()-start, "peak_torch_allocated_mib": torch.cuda.max_memory_allocated()/2**20})


def infer(root):
    import torch
    from wmfracture.belief_network import BeliefCorrection
    settings = json.loads((root/"belief-settings.json").read_text())
    evidence = load_evidence(root, "test")
    manifest = json.loads((root/"belief-trained/manifest.json").read_text())
    prep = root/"belief-trained/preprocessing.npz"
    if digest(prep) != manifest["preprocessing_sha256"]:
        raise ValueError("Belief preprocessing changed")
    with np.load(prep, allow_pickle=False) as archive:
        prior, temperature, scale = archive["prior"], float(archive["temperature"]), float(archive["information_scale"])
        center, components = archive["pca_center"], archive["pca_components"]
    accepted, scores = evidence["accepted"], evidence["scores"]
    uniform = np.zeros_like(scores)
    valid = accepted.any(1)
    uniform[valid] = accepted[valid]/accepted[valid].sum(1, keepdims=True)
    hard = np.zeros_like(scores)
    hard[valid, np.argmin(np.where(accepted[valid], scores[valid], np.inf), axis=1)] = 1
    names = ["uniform_accepted", "rgbd_map", "calibrated_energy"]
    all_weights = [uniform, hard, energy_weights(scores, accepted, prior, temperature)]
    embeddings = np.load(root/"frozen-predictor-features.npy", allow_pickle=False)
    for record in manifest["models"]:
        path = root/"belief-trained"/record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError("Belief checkpoint changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        values = vector(record["model"], evidence, embeddings, center, components)
        values = (values-checkpoint["mean"].numpy())/checkpoint["std"].numpy()
        model = BeliefCorrection(checkpoint["input_size"], settings["hidden_width"], settings["residual_logit_bound"]).cuda().eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        with torch.inference_mode():
            correction = model(torch.as_tensor(values, device="cuda"), torch.as_tensor(gate(scores, scale), device="cuda")).cpu().numpy()
        all_weights.append(energy_weights(scores, accepted, prior, temperature, correction))
        names.append(f"{record['model']}/seed-{record['seed']}")
    output = root/"posterior"
    output.mkdir()
    weights = np.stack(all_weights)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights[:, valid].sum(-1), 1):
        raise ValueError("Invalid state weights")
    np.savez_compressed(output/"weights.npz", weights=weights, prefix_index=evidence["prefix_index"])
    packet = {"schema": "weighted-reachable-cohesive-state-packet-1", "models": names,
        "candidate_state_library": "candidate-prefix-states.json", "state_library_sha256": digest(root/"candidate-prefix-states.json"),
        "weights_sha256": digest(output/"weights.npz"), "belief_input_sha256": digest(root/"belief-inputs/test.npz"),
        "pretrained_feature_sha256": digest(root/"frozen-predictor-features.npy"), "training_manifest_sha256": digest(root/"belief-trained/manifest.json"),
        "future_plans_read": False, "test_labels_read": False, "candidate_states_averaged": False,
        "valid_prefix_count": int(valid.sum()), "prefix_count": len(valid),
        "static_prior_max_weight": float(prior.max())}
    write_json(output/"packet.json", packet)
    (output/"packet.sha256").write_text(digest(output/"packet.json"))
    print(f"BELIEFS_SEALED methods={len(names)} prefixes={len(valid)}", flush=True)


def prepare(project, root, experiment):
    data = json.loads((project/experiment["base_data_config"]).read_text())
    data.update({key: experiment[key] for key in ("seed", "history_counts", "history_peak_min_m", "history_peak_max_m")})
    data["write_belief_data"] = True
    data["scope"] = experiment["scope"]
    write_json(root/"config.json", data)
    write_json(root/"experiment.json", experiment)
    write_json(root/"belief-settings.json", {key: experiment[key] for key in SETTING_KEYS})
    shutil.copy2(project/"config/world-model.json", root/"pretrained-config.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "infer", "continue", "forecast", "evaluate", "branch", "render"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--prefix", type=int)
    parser.add_argument("--impact", type=float)
    parser.add_argument("--impact-time", type=float, default=.036)
    parser.add_argument("--model", default="calibrated_energy")
    args = parser.parse_args()
    project = Path(os.environ["WM_PROJECT_ROOT"])
    if args.stage in ("train", "infer"):
        globals()[args.stage](args.run); return
    if args.stage in ("forecast", "evaluate", "branch", "render"):
        if args.stage == "evaluate":
            from belief_state_evaluate import evaluate
            evaluate(args.run); return
        command = ["bash", str(project/"scripts/wsl.sh"), "python", str(project/"scripts/belief_state_forecast.py"), "--run", str(args.run), "--stage", args.stage]
        if args.stage == "branch":
            if args.prefix is None or args.impact is None:
                parser.error("branch requires --prefix and --impact")
            command += ["--prefix", str(args.prefix), "--impact", str(args.impact), "--impact-time", str(args.impact_time), "--model", args.model]
        launch(command, args.run/f"{args.stage}-manual.log"); return
    if args.run:
        root = args.run
        experiment = json.loads((root/"experiment.json").read_text())
        # Early generated data may predate the pruned inference settings file.
        write_json(root/"belief-settings.json", {key: experiment[key] for key in SETTING_KEYS})
    else:
        experiment = json.loads((project/"config/belief-state.json").read_text())
        root = Path(os.environ["WM_RESEARCH_ROOT"])/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-belief-state")
        root.mkdir(parents=True)
        prepare(project, root, experiment)
        print(f"OUTPUT={root}", flush=True)
        launch(["bash", str(project/"scripts/wsl.sh"), "python", str(project/"scripts/action_adapter_data.py"), "--run", str(root)], root/"generate.log")
    write_json(root/"code-sha256.json", {str(path.relative_to(project)): digest(path) for path in [Path(__file__),
        project/"scripts/action_adapter_data.py", project/"scripts/belief_state_forecast.py", project/"scripts/belief_state_evaluate.py",
        project/"wmfracture/belief_state.py", project/"wmfracture/belief_network.py", project/"wmfracture/cohesive_batch.py"]})
    launch([sys.executable, "-u", str(project/"scripts/action_adapter.py"), "--stage", "features", "--run", str(root)], root/"features.log")
    for stage in ("train", "infer", "forecast", "evaluate", "render"):
        launch([sys.executable, "-u", str(Path(__file__)), "--stage", stage, "--run", str(root)], root/f"{stage}.log")


if __name__ == "__main__":
    main()
