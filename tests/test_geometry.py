import numpy as np
import torch

from wmfracture.geometry import advect_gaussians, make_transform, rigid_alignment_rms, stratified_carriers, unpack_covariance


def test_normalization_rotation_roundtrip():
    points = torch.tensor([[-2., 1., 3.], [4., -1., 2.], [1., 2., -3.]])
    rotation, scale, origin, center = make_transform(points, [25, -40, 15], 1.2, [1, 1, 1])
    transformed = (points @ rotation.T - origin) * scale + center
    recovered = ((transformed - center) / scale + origin) @ rotation
    torch.testing.assert_close(recovered, points)
    torch.testing.assert_close((transformed.amax(0) - transformed.amin(0)).max(), torch.tensor(1.2))


def test_carrier_volume_independent_of_sampling_density():
    points = np.array([[.41, .41, .41], [.42, .42, .42], [.43, .43, .43], [.61, .41, .41]])
    for per_cell in (1, 2, 4):
        selected, volumes, cells = stratified_carriers(points, 10, 1, per_cell, 42)
        assert cells == 2
        np.testing.assert_allclose(volumes.sum(), .002)
        np.testing.assert_array_equal(selected, stratified_carriers(points, 10, 1, per_cell, 42)[0])


def test_affine_skinning_matches_global_affine_map_and_preserves_spd():
    rest_carriers = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    binding = torch.tensor([0, 1, 1])
    offsets = torch.tensor([[.1, .2, 0.], [-.1, .3, .1], [0., 0., .2]])
    affine = torch.tensor([[1.2, .3, 0.], [0., .8, .1], [0., 0., 1.1]])
    translation = torch.tensor([.2, -.3, .5])
    covariance = torch.diag(torch.tensor([.01, .02, .03])).repeat(3, 1, 1)
    positions, packed = advect_gaussians(rest_carriers @ affine.T + translation, affine.repeat(2, 1, 1),
                                       binding, offsets, covariance)
    expected = (rest_carriers[binding] + offsets) @ affine.T + translation
    torch.testing.assert_close(positions, expected)
    transformed_covariance = unpack_covariance(packed)
    torch.testing.assert_close(transformed_covariance, affine @ covariance @ affine.T)
    assert (torch.linalg.eigvalsh(transformed_covariance) > 0).all()


def test_shape_change_excludes_rigid_rotation_and_translation():
    points = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = points @ rotation + torch.tensor([4., -2., .5])
    assert rigid_alignment_rms(points, moved) < 1e-6
    assert rigid_alignment_rms(points, moved * 1.5) > 0.1
