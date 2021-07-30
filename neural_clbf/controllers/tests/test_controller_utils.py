"""Test some controller utilities"""
import torch

from neural_clbf.controllers.controller_utils import normalize_with_angles
from neural_clbf.systems.tests.mock_system import MockSystem


def test_normalize_x():
    """Test the ability to normalize states"""
    # Define the model system
    params = {}
    system = MockSystem(params)

    # Define states on which to test.
    # Start with the upper and lower state limits
    x_upper, x_lower = system.state_limits
    x_upper = x_upper.unsqueeze(0)
    x_lower = x_lower.unsqueeze(0)

    # These should be normalized so that the first dimension becomes 1 and -1 (resp)
    # The second dimension is an angle and should be replaced with its sine and cosine
    x_upper_norm = normalize_with_angles(system, x_upper)
    assert torch.allclose(x_upper_norm[0, 0], torch.ones(1))
    assert torch.allclose(
        x_upper_norm[0, 1:],
        torch.tensor([torch.sin(x_upper[0, 1]), torch.cos(x_upper[0, 1])]),
    )
    x_lower_norm = normalize_with_angles(system, x_lower)
    assert torch.allclose(x_lower_norm[0, 0], -torch.ones(1))
    assert torch.allclose(
        x_lower_norm[0, 1:],
        torch.tensor([torch.sin(x_lower[0, 1]), torch.cos(x_lower[0, 1])]),
    )

    # Also test that the center of the range is normalized to zero
    x_center = 0.5 * (x_upper + x_lower)
    x_center_norm = normalize_with_angles(system, x_center)
    assert torch.allclose(x_center_norm[0, 0], torch.zeros(1))
    assert torch.allclose(
        x_center_norm[0, 1:],
        torch.tensor([torch.sin(x_center[0, 1]), torch.cos(x_center[0, 1])]),
    )
