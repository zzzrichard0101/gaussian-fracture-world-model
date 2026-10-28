"""Coordinate and carrier operations independent of the upstream solver."""
import numpy as np
import torch


def unpack_covariance(packed):
    matrix = packed.new_zeros((len(packed), 3, 3))
    matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2] = packed[:, 0], packed[:, 1], packed[:, 2]
    matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2] = packed[:, 1], packed[:, 3], packed[:, 4]
    matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2] = packed[:, 2], packed[:, 4], packed[:, 5]
    return matrix


def pack_covariance(matrix):
    return torch.stack([matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2],
                        matrix[:, 1, 1], matrix[:, 1, 2], matrix[:, 2, 2]], dim=1).contiguous()


def make_transform(points, degrees, extent, simulation_center):
    rotation = torch.eye(3, dtype=points.dtype, device=points.device)
    for axis, degree in enumerate(degrees):
        angle = np.deg2rad(degree)
        c, s = np.cos(angle), np.sin(angle)
        if axis == 0:
            values = [[1, 0, 0], [0, c, -s], [0, s, c]]
        elif axis == 1:
            values = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        else:
            values = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        rotation = points.new_tensor(values) @ rotation
    rotated = points @ rotation.T
    minimum, maximum = rotated.amin(dim=0), rotated.amax(dim=0)
    scale = float(extent / (maximum - minimum).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Asset must have finite nonzero extent")
    origin = (minimum + maximum) / 2
    center = points.new_tensor(simulation_center)
    return rotation, scale, origin, center


def stratified_carriers(points, grid, domain_size, per_cell, seed):
    """Sample source Gaussians per occupied MPM cell; preserve cell volume.

    This follows the upstream occupied-cell volume approximation. It does not
    infer a watertight solid or fill cells with no Gaussian centers.
    """
    dx = domain_size / grid
    indices = np.floor(np.asarray(points) / dx).astype(np.int64)
    if (indices < 1).any() or (indices >= grid - 2).any():
        raise ValueError("Initial particles need a safe margin inside the MPM domain")
    keys = (indices[:, 0] * grid + indices[:, 1]) * grid + indices[:, 2]
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(points)), keys))
    _, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    rank = np.arange(len(order)) - np.repeat(starts, counts)
    selected = order[rank < per_cell]
    _, inverse, counts_selected = np.unique(keys[selected], return_inverse=True, return_counts=True)
    volumes = np.asarray(dx ** 3 / counts_selected[inverse], dtype=np.float32)
    return selected, volumes, len(starts)


def advect_gaussians(carrier_positions, carrier_affines, binding, rest_offsets, rest_covariance):
    """Nearest-carrier local affine skinning; no blending across carriers.

    Affines may be elastic/trial F (legacy) or an accumulated kinematic map.
    Even the latter is a local continuum approximation: nearest-carrier skinning
    has limitations near carrier boundaries and cracks, and is not a validated
    fracture-aware transfer method.
    """
    affines = carrier_affines[binding]
    positions = carrier_positions[binding] + (affines @ rest_offsets[..., None]).squeeze(-1)
    covariance = affines @ rest_covariance @ affines.transpose(1, 2)
    return positions.contiguous(), pack_covariance(covariance)


def rigid_alignment_rms(initial, current):
    """Shape change remaining after best rigid translation and rotation."""
    rest = initial - initial.mean(dim=0)
    deformed = current - current.mean(dim=0)
    u, _, vh = torch.linalg.svd(rest.T @ deformed)
    correction = torch.eye(3, dtype=rest.dtype, device=rest.device)
    correction[2, 2] = torch.linalg.det(u @ vh)
    rotation = u @ correction @ vh
    return (deformed - rest @ rotation).square().sum(dim=1).mean().sqrt()
