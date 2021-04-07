"""Test the neural CLBF controller with system identification"""
import torch

from neural_clbf.controllers.neural_sid_clbf_controller import (
    NeuralSIDCLBFController,
)
from neural_clbf.systems.tests.mock_system import MockSystem
from neural_clbf.experiments.common.episodic_datamodule import EpisodicDataModule


def test_init_neuralrclbfcontroller():
    """Test the initialization of a NeuralCLBFController"""
    # Define the model system
    params = {}
    system = MockSystem(params)
    # Define the datamodule
    initial_domain = [
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    dm = EpisodicDataModule(system, initial_domain)

    # Instantiate with a list of only one scenarios
    scenarios = [params]
    controller = NeuralSIDCLBFController(system, scenarios, dm)
    assert controller is not None


def test_normalize_x():
    """Test the ability to normalize states"""
    # Define the model system
    params = {}
    system = MockSystem(params)
    # Define the datamodule
    initial_domain = [
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    dm = EpisodicDataModule(system, initial_domain)

    # Instantiate with a list of only one scenarios
    scenarios = [params]
    controller = NeuralSIDCLBFController(system, scenarios, dm)

    # Define states on which to test.
    # Start with the upper and lower state limits
    x_upper, x_lower = system.state_limits
    x_upper = x_upper.unsqueeze(0)
    x_lower = x_lower.unsqueeze(0)

    # These should be normalized so that the first dimension becomes 1 and -1 (resp)
    # The second dimension is an angle and should be replaced with its sine and cosine
    x_upper_norm = controller.normalize_w_angles(x_upper)
    assert torch.allclose(x_upper_norm[0, 0], torch.ones(1))
    assert torch.allclose(
        x_upper_norm[0, 1:],
        torch.tensor([torch.sin(x_upper[0, 1]), torch.cos(x_upper[0, 1])]),
    )
    x_lower_norm = controller.normalize_w_angles(x_lower)
    assert torch.allclose(x_lower_norm[0, 0], -torch.ones(1))
    assert torch.allclose(
        x_lower_norm[0, 1:],
        torch.tensor([torch.sin(x_lower[0, 1]), torch.cos(x_lower[0, 1])]),
    )

    # Also test that the center of the range is normalized to zero
    x_center = 0.5 * (x_upper + x_lower)
    x_center_norm = controller.normalize_w_angles(x_center)
    assert torch.allclose(x_center_norm[0, 0], torch.zeros(1))
    assert torch.allclose(
        x_center_norm[0, 1:],
        torch.tensor([torch.sin(x_center[0, 1]), torch.cos(x_center[0, 1])]),
    )
