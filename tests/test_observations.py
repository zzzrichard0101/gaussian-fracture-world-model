import numpy as np
import pytest
import torch

from wmfracture.observations import GaussianRGBD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Gaussian observation integration")
def test_camera_depth_and_floor_occlusion_in_same_compositing():
    renderer = GaussianRGBD({"width": 32, "camera_eye": [0., -3., 1.],
                            "camera_target": [0., 0., 1.], "vertical_fov_degrees": 45.,
                            "background": [1., 1., 1.]})
    points = torch.tensor([[0., 0., 1.]], device="cuda")
    cov = torch.tensor([[.04, 0., 0., .04, 0., .04]], device="cuda")
    opacity = torch.tensor([[.99]], device="cuda")
    color = torch.tensor([[1., 0., 0.]], device="cuda")
    result = renderer.points(points, cov, color, opacity, 1)
    mask = result["object_alpha"] > .5
    assert mask.any()
    np.testing.assert_allclose(result["depth"][mask], 3., atol=1e-5)
    # A closer opaque non-object Gaussian must occlude the object mask.
    combined = renderer.points(torch.cat([points, torch.tensor([[0., -.7, 1.]], device="cuda")]),
                               cov.repeat(2, 1), color.repeat(2, 1), opacity.repeat(2, 1), 1)
    assert combined["object_alpha"][16, 16] < result["object_alpha"][16, 16] * .1
    assert combined["depth"][16, 16] < result["depth"][16, 16]
