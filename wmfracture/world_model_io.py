"""Shared array/mask contracts without pretrained-code or private-target access."""
import numpy as np


def causal_indices(config):
    size, patch = config["image_size"], config["patch_size"]
    slots, context = config["observation_slots"], config["context_slots"]
    if size % patch or not 0 < context < slots or config["frames_per_slot"] != config["tubelet_size"]:
        raise ValueError("Context boundary must align with non-overlapping tubelets")
    per_slot = (size // patch) ** 2
    return np.arange(context * per_slot, dtype=np.int64), np.arange(context * per_slot, slots * per_slot, dtype=np.int64)


def context_canvas(prefix, config, filler=0.):
    prefix = np.asarray(prefix, dtype=np.float32)
    size = config["image_size"]
    expected = (config["context_slots"], size, size, 3)
    if prefix.shape != expected or not np.isfinite(prefix).all():
        raise ValueError("Only the declared observed prefix may be supplied")
    canvas = np.full((config["observation_slots"], size, size, 3), filler, dtype=np.float32)
    canvas[:len(prefix)] = prefix
    return np.repeat(canvas, config["frames_per_slot"], axis=0)


def validate_rank_arrays(arrays, config):
    if set(arrays) != {"context_rgb", "candidate_rgb", "context_roi", "prefix_metrics"}:
        raise ValueError("Unexpected ranking input, possibly private target data")
    prefix = arrays["context_rgb"]
    context_canvas(prefix, config)
    clips = arrays["candidate_rgb"]
    expected_tail = (config["observation_slots"], config["image_size"], config["image_size"], 3)
    if clips.ndim != 5 or clips.shape[1:] != expected_tail or clips.shape[0] == 0:
        raise ValueError("Invalid candidate video shape")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite ranking inputs")
    if not np.array_equal(clips[:, :config["context_slots"]], np.broadcast_to(prefix, clips[:, :config["context_slots"]].shape)):
        raise ValueError("Candidate clips must share exactly the observed context")
    return arrays


def hybrid_choice(rgbd_accepted, latent_energies):
    if not rgbd_accepted:
        return None
    if not np.isfinite(latent_energies).all():
        raise ValueError("Nonfinite learned-prior energy")
    return min(rgbd_accepted, key=lambda index: (latent_energies[index], index))
