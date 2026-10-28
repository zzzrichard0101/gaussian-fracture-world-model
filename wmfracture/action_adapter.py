"""Small supervised fixed-horizon dynamics heads; pretrained model stays frozen."""
import numpy as np
import torch
from torch import nn


class DynamicsHead(nn.Module):
    def __init__(self, input_size, horizon, width, future_action_column=None):
        super().__init__()
        self.future_action_column = future_action_column
        self.network = nn.Sequential(nn.Linear(input_size, width), nn.SiLU(), nn.Linear(width, width),
                                     nn.SiLU(), nn.Linear(width, horizon+1))

    def forward(self, inputs):
        values = self.network(inputs)
        if self.future_action_column is None:
            return values
        # Position is observed BEFORE the impulse at target_times[0]. The
        # future velocity increment may affect only later positions.
        before_impulse = inputs.clone()
        before_impulse[:, self.future_action_column] = 0.
        boundary_position = self.network(before_impulse)[:, :1]
        return torch.cat([boundary_position, values[:, 1:]], dim=1)


def fit_pca(embeddings, training_prefix_indices, dimensions):
    training = embeddings[np.unique(training_prefix_indices)].astype(np.float64)
    center = training.mean(0)
    _, _, vt = np.linalg.svd(training-center, full_matrices=False)
    return center.astype(np.float32), vt[:dimensions].astype(np.float32)


def model_inputs(kind, arrays, embeddings, pca_center, pca_components):
    actions = arrays["actions"].astype(np.float32)
    if kind == "action_only":
        return actions
    blocks = [arrays["rgbd_features"].astype(np.float32)]
    if "vjepa" in kind:
        blocks.append((embeddings[arrays["prefix_index"]]-pca_center) @ pca_components.T)
    blocks.append(actions[:, :1] if kind.endswith("no_future_action") else actions)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def training_normalizer(values):
    return values.mean(0).astype(np.float32), np.maximum(values.std(0), 1e-6).astype(np.float32)
