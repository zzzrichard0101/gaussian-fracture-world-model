"""Resume sealed state hypotheses under new plans, then Gaussian-render MAPs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from wmfracture.cohesive import InterfaceLaw, InterfaceState, TwoBlockSystem
from wmfracture.cohesive_batch import batch_rollout


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def packet(root):
    path = root/"posterior/packet.json"
    if digest(path) != (root/"posterior/packet.sha256").read_text():
        raise ValueError("State packet seal changed")
    meta = json.loads(path.read_text())
    if digest(root/meta["candidate_state_library"]) != meta["state_library_sha256"] or digest(root/"posterior/weights.npz") != meta["weights_sha256"]:
        raise ValueError("Candidate states/weights changed after sealing")
    if digest(root/"belief-inputs/test.npz") != meta["belief_input_sha256"]:
        raise ValueError("Sealed observation-to-state mapping changed")
    library = json.loads((root/meta["candidate_state_library"]).read_text())
    with np.load(root/"posterior/weights.npz", allow_pickle=False) as archive:
        weights, prefixes = archive["weights"], archive["prefix_index"]
    return meta, library, weights, prefixes


def sealed_probe(root, prefix):
    # This archive is bound by packet(), unlike editable display metadata.
    with np.load(root/"belief-inputs/test.npz", allow_pickle=False) as archive:
        positions = np.flatnonzero(archive["prefix_index"] == prefix)
        if len(positions) != 1:
            raise ValueError("Unknown sealed observation prefix")
        return float(archive["probe"][positions[0], 0])


def candidate_states(library, probe):
    model = library["model"]
    selected = [record for record in library["candidates"] if np.isclose(record["probe"], probe, atol=1e-8, rtol=0)]
    if len(selected) != 13:
        raise ValueError("Missing cutoff state hypotheses")
    states = []
    for record in selected:
        state = InterfaceState(**record["state"])
        TwoBlockSystem(InterfaceLaw(**model["law"]), state, model["mass_left"], model["mass_right"], model["width"])
        if state.maximum_opening < record["history_peak"]-1e-12:
            raise ValueError("Pre-observation history was erased")
        states.append(state)
    return states


def forecast(root):
    meta, library, _, _ = packet(root)
    config = json.loads((root/"config.json").read_text())
    plans = json.loads((root/"physics-bank-index.json").read_text())
    states, impacts = [], []
    for plan in plans:
        states.extend(candidate_states(library, plan["probe"]))
        impacts.extend([plan["impact"]]*13)
    result = batch_rollout(library["model"], states, np.zeros(len(states)), impacts, config["candidate_dt"],
        config["target_times"], future_time=config["future_time"], start_elapsed=library["observation_end"])
    gaps = result["opening"].reshape(len(plans), 13, -1)
    failed = (result["damage"][:, -1] == 1).reshape(len(plans), 13)
    # Candidate-only replay check, not access to heldout test responses.
    with np.load(root/"physics-bank.npz", allow_pickle=False) as archive:
        difference = float(np.abs(gaps-archive["gaps"]).max())
        same_events = bool(np.array_equal(failed, archive["failure"]))
    if difference > 1e-8 or not same_events:
        raise ValueError("Cutoff-state continuation disagrees with uninterrupted candidate replay")
    output = root/"forecasts"
    output.mkdir()
    np.savez_compressed(output/"candidate-branches.npz", gaps=gaps, failure=failed)
    audit = {"posterior_packet_sha256": digest(root/"posterior/packet.json"),
        "future_plan_index_sha256": digest(root/"physics-bank-index.json"),
        "future_test_case_index_sha256": digest(root/"inputs/test.npz"),
        "forecast_sha256": digest(output/"candidate-branches.npz"), "resumed_candidate_branches": len(states),
        "maximum_difference_from_uninterrupted_candidate_m": difference, "same_failure_events": same_events,
        "damage_monotone_at_samples": bool((np.diff(result["maximum_opening"], axis=1) >= 0).all()),
        "test_labels_read": False, "weights_changed_for_future_plan": False, "states_averaged": False}
    write_json(output/"seal.json", audit)
    (output/"seal.sha256").write_text(digest(output/"seal.json"))
    print(f"FORECAST_SEALED branches={len(states)} resume_error={difference:g}m", flush=True)


def branch(root, prefix, impact, impact_time, name, identifier=None):
    meta, library, weights, prefixes = packet(root)
    if impact_time <= library["observation_end"] or not np.isfinite([impact, impact_time]).all():
        raise ValueError("New action must be finite and after the sealed observation cutoff")
    config = json.loads((root/"config.json").read_text())
    if impact_time >= max(config["target_times"]):
        raise ValueError("New action outside the forecast horizon")
    position = np.flatnonzero(prefixes == prefix)
    if len(position) != 1 or name not in meta["models"]:
        raise ValueError("Unknown sealed prefix/model")
    q = weights[meta["models"].index(name), position[0]]
    if not np.isclose(q.sum(), 1):
        raise ValueError("Cannot branch an empty observation match")
    states = candidate_states(library, sealed_probe(root, prefix))
    result = batch_rollout(library["model"], states, np.zeros(13), np.full(13, impact), config["candidate_dt"],
        config["target_times"], future_time=impact_time, start_elapsed=library["observation_end"])
    output = root/"counterfactuals"
    output.mkdir(exist_ok=True)
    stem = identifier or f"prefix-{prefix}-{name.replace('/', '_')}-dv-{impact:g}-t-{impact_time:g}"
    np.savez_compressed(output/f"{stem}.npz", gaps=result["opening"], failure=result["damage"][:, -1] == 1,
        weights=q, x_left=result["x_left"], x_right=result["x_right"])
    record = {"prefix": prefix, "model": name, "impact": impact, "impact_time": impact_time,
        "map_candidate": int(np.argmax(q)), "failure_weight": float(np.clip(q@(result["damage"][:, -1] == 1), 0, 1)),
        "posterior_packet_sha256": digest(root/"posterior/packet.json"),
        "weights_sha256": hashlib.sha256(q.tobytes()).hexdigest(), "file": stem+".npz",
        "test_reference_available": False, "note": "Prediction-only branch from unchanged sealed states; not evaluated against a real observation."}
    write_json(output/f"{stem}.json", record)
    print(f"BRANCH {stem} failure_weight={record['failure_weight']:.4f}", flush=True)
    return record


def render(root):
    from cohesive_probe import BlockObservations
    from PIL import Image, ImageDraw
    experiment = json.loads((root/"experiment.json").read_text())
    config = json.loads((root/"config.json").read_text())
    _, library, _, prefixes = packet(root)
    prefix = next(int(i) for i in prefixes if sealed_probe(root, int(i)) > 0)
    branches = [branch(root, prefix, row["impact"], row["impact_time"], experiment["demo_model"], row["id"])
                for row in experiment["custom_branches"]]
    if len({record["weights_sha256"] for record in branches}) != 1:
        raise ValueError("Counterfactual plans changed inferred state weights")
    observer = BlockObservations(library["model"])
    width = config["render_width"]
    frames = root/"counterfactual-frames"
    frames.mkdir(exist_ok=True)
    trajectories = [np.load(root/"counterfactuals"/record["file"], allow_pickle=False) for record in branches]
    for frame_index, elapsed in enumerate(config["target_times"]):
        canvas = Image.new("RGB", (width*len(branches), width+80), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (record, values) in enumerate(zip(branches, trajectories)):
            candidate = record["map_candidate"]
            rgb = observer.capture({"x_left": values["x_left"][candidate, frame_index],
                                    "x_right": values["x_right"][candidate, frame_index]})["rgb"]
            canvas.paste(Image.fromarray(np.round(np.clip(rgb, 0, 1)*255).astype(np.uint8)), (column*width, 0))
            draw.text((column*width+7, width+4), f"dv={record['impact']:.3f}m/s at t={record['impact_time']:.3f}s", fill="black")
            draw.text((column*width+7, width+21), f"Final separation weight={record['failure_weight']:.3f}", fill="black")
            draw.text((column*width+7, width+38), f"Same belief; physical MAP #{candidate}", fill="black")
        draw.text((7, width+59), f"t={elapsed:.3f}s | Predictions only, 1D fixture, slow playback", fill="black")
        canvas.save(frames/f"{frame_index:05d}.png")
    canvas.save(root/"counterfactual-end.png")
    for values in trajectories:
        values.close()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "15", "-i", str(frames/"%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(root/"counterfactuals.mp4")], check=True)
    write_json(root/"counterfactual-audit.json", {"branches": branches, "same_weights_for_all_plans": True,
        "rendered_trajectory": "One valid physical MAP component, never the mixture mean",
        "demo_rule": "First probed test prefix, learned_vjepa seed 17; three plans fixed before evaluation"})
    target = Path(os.environ["WM_PROJECT_ROOT"])/"artifacts/belief-state"
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("counterfactuals.mp4", "counterfactual-end.png", "counterfactual-audit.json"):
        shutil.copy2(root/filename, target/filename)
    print(f"GAUSSIAN_VIDEO={root/'counterfactuals.mp4'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=["forecast", "branch", "render"], required=True)
    parser.add_argument("--prefix", type=int)
    parser.add_argument("--impact", type=float)
    parser.add_argument("--impact-time", type=float, default=.036)
    parser.add_argument("--model", default="calibrated_energy")
    args = parser.parse_args()
    if args.stage == "branch":
        branch(args.run, args.prefix, args.impact, args.impact_time, args.model)
    else:
        globals()[args.stage](args.run)
