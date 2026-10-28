"""Frozen V-JEPA 2 encoder + predictor, strict future-masked latent inference."""
import copy
import gc
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from wmfracture.world_model_io import causal_indices, context_canvas


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VJEPAWorld:
    def __init__(self, config):
        from src.hub.backbones import vjepa2_vit_large
        self.config = config
        source = Path(os.environ["WM_VJEPA_ROOT"])
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != config["source_commit"]:
            raise ValueError("V-JEPA source revision mismatch")
        path = Path(os.environ["WM_RESEARCH_ROOT"]) / config["weights_relative_path"]
        if path.stat().st_size != config["weights_bytes"] or file_sha256(path) != config["weights_sha256"]:
            raise ValueError("Official checkpoint checksum/size mismatch")
        torch.set_num_threads(8)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.encoder, self.predictor = vjepa2_vit_large(pretrained=False)
        self.teacher = copy.deepcopy(self.encoder)
        self.load_audit = {}
        for name, network, key in [("encoder", self.encoder, "encoder"), ("predictor", self.predictor, "predictor"),
                                    ("target_encoder", self.teacher, "target_encoder")]:
            weights = {key.replace("module.", "").replace("backbone.", ""): value for key, value in checkpoint[key].items()}
            result = network.load_state_dict(weights, strict=False)
            allowed_buffers = {"pos_embed", "predictor_pos_embed"}
            if result.missing_keys or set(result.unexpected_keys) - allowed_buffers:
                raise ValueError(f"Incomplete pretrained {name}: {result}")
            network.requires_grad_(False).eval().cuda()
            self.load_audit[name] = {"checkpoint_fields": len(weights), "missing_keys": result.missing_keys,
                                     "ignored_position_buffers": result.unexpected_keys,
                                     "parameters": sum(parameter.numel() for parameter in network.parameters())}
        del checkpoint, weights
        gc.collect()
        context, target = causal_indices(config)
        self.context = torch.from_numpy(context).cuda()[None]
        self.target = torch.from_numpy(target).cuda()[None]
        self.mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None, None]
        self.std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None, None]

    def tensor(self, frames):
        values = torch.as_tensor(np.ascontiguousarray(frames), device="cuda", dtype=torch.float32).permute(3, 0, 1, 2)[None]
        return (values - self.mean) / self.std

    @torch.inference_mode()
    def predict(self, prefix, filler=0.):
        values = self.tensor(context_canvas(prefix, self.config, filler))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context = self.encoder(values, masks=[self.context])
            prediction = self.predictor(context, [self.context], [self.target], mask_index=self.config["predictor_mask_index"])
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite pretrained predictor output")
        return prediction.float()

    @torch.inference_mode()
    def persistence(self, prefix):
        values = self.tensor(context_canvas(prefix, self.config))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context = self.teacher(values, masks=[self.context])
        context = F.layer_norm(context.float(), (context.shape[-1],))
        per_slot = (self.config["image_size"] // self.config["patch_size"]) ** 2
        return context[:, -per_slot:].repeat(1, self.config["observation_slots"] - self.config["context_slots"], 1)

    @torch.inference_mode()
    def encode_targets(self, clip):
        if len(clip) != self.config["observation_slots"]:
            raise ValueError("Wrong target clip length")
        values = self.tensor(np.repeat(clip, self.config["frames_per_slot"], axis=0))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = self.teacher(values)
        encoded = F.layer_norm(encoded.float(), (encoded.shape[-1],))
        return encoded[:, self.target[0]]

    def roi_weights(self, context_roi):
        size = self.config["image_size"] // self.config["patch_size"]
        mask = torch.tensor(context_roi, dtype=torch.float32, device="cuda")[None, None]
        mask = F.adaptive_max_pool2d(mask, (size, size))
        radius = self.config["roi_dilation_patches"]
        if radius:
            mask = F.max_pool2d(mask, radius*2+1, stride=1, padding=radius)
        if mask.sum() == 0:
            raise ValueError("No foreground support in observed prefix")
        return mask.flatten().repeat(self.config["observation_slots"] - self.config["context_slots"])

    @staticmethod
    def energy(predicted, target, roi):
        per_token = (predicted - target).abs().mean(-1)[0]
        return {"full_mae": float(per_token.mean()), "roi_mae": float((per_token * roi).sum() / roi.sum())}
