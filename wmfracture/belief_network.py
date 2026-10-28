"""Bounded corrections to observation evidence; no future-action argument."""
import torch
from torch import nn


class BeliefCorrection(nn.Module):
    def __init__(self, input_size, width, bound=1.):
        super().__init__()
        self.bound = bound
        self.network = nn.Sequential(nn.Linear(input_size, width), nn.SiLU(), nn.Linear(width, 13))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observed_features, information_gate):
        return self.bound*self.network(observed_features).tanh()*information_gate[:, None]
