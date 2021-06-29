"""Test the the vanilla neural clbf controller"""
import torch
import random

from torch.autograd.functional import jacobian

from neural_clbf.controllers.neural_clbf_controller import (
    NeuralCLBFController,
)
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.systems.tests.mock_system import MockSystem
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule


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

    experiment_suite = ExperimentSuite([])

    # Instantiate with a list of only one scenarios
    scenarios = [params]
    controller = NeuralCLBFController(system, scenarios, dm, experiment_suite)
    assert controller is not None
    assert controller.controller_period > 0


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

    experiment_suite = ExperimentSuite([])

    # Instantiate with a list of only one scenarios
    scenarios = [params]
    controller = NeuralCLBFController(system, scenarios, dm, experiment_suite)

    # Define states on which to test.
    # Start with the upper and lower state limits
    x_upper, x_lower = system.state_limits
    x_upper = x_upper.unsqueeze(0)
    x_lower = x_lower.unsqueeze(0)

    # These should be normalized so that the first dimension becomes 1 and -1 (resp)
    # The second dimension is an angle and should be replaced with its sine and cosine
    x_upper_norm = controller.normalize_with_angles(x_upper)
    assert torch.allclose(x_upper_norm[0, 0], torch.ones(1))
    assert torch.allclose(
        x_upper_norm[0, 1:],
        torch.tensor([torch.sin(x_upper[0, 1]), torch.cos(x_upper[0, 1])]),
    )
    x_lower_norm = controller.normalize_with_angles(x_lower)
    assert torch.allclose(x_lower_norm[0, 0], -torch.ones(1))
    assert torch.allclose(
        x_lower_norm[0, 1:],
        torch.tensor([torch.sin(x_lower[0, 1]), torch.cos(x_lower[0, 1])]),
    )

    # Also test that the center of the range is normalized to zero
    x_center = 0.5 * (x_upper + x_lower)
    x_center_norm = controller.normalize_with_angles(x_center)
    assert torch.allclose(x_center_norm[0, 0], torch.zeros(1))
    assert torch.allclose(
        x_center_norm[0, 1:],
        torch.tensor([torch.sin(x_center[0, 1]), torch.cos(x_center[0, 1])]),
    )


def test_V_jacobian():
    """Test computation of Lie Derivatives"""
    # Set a random seed for repeatability
    random.seed(0)
    torch.manual_seed(0)

    # Create the controller
    params = {}
    system = MockSystem(params)
    scenarios = [params]
    # Define the datamodule
    initial_domain = [
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    dm = EpisodicDataModule(system, initial_domain)
    experiment_suite = ExperimentSuite([])
    controller = NeuralCLBFController(
        system,
        scenarios,
        dm,
        experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
    )

    # Create the points and perturbations with which to test the Jacobian
    N_test = 10
    x = torch.Tensor(N_test, system.n_dims).uniform_(-1.0, 1.0)
    dx = torch.Tensor(N_test, system.n_dims).uniform_(1e-3, 2e-3)

    # Compute V and its Jacobian
    V, gradV = controller.V_with_jacobian(x)
    # and use these to get the expected change
    deltaV_expected = torch.bmm(gradV, dx.unsqueeze(-1))

    # Compare first with the autodiff Jacobian
    J_V_x = torch.zeros(N_test, 1, x.shape[1])
    for i in range(N_test):
        J_V_x[i, :, :] = jacobian(controller.V, x[i, :].unsqueeze(0))

    assert torch.allclose(gradV.squeeze(), J_V_x.squeeze())

    # To validate the Jacobian, approximate with finite difference
    x_next = x + dx
    V_next = controller.V(x_next)
    deltaV_simulated = V_next - V

    # Make sure they're close
    assert torch.allclose(
        deltaV_expected.squeeze(), deltaV_simulated.squeeze(), rtol=1e-2
    )


def test_V_lie_derivatives():
    """Test computation of Lie Derivatives"""
    # Set a random seed for repeatability
    random.seed(0)
    torch.manual_seed(0)

    # Create the controller
    params = {}
    system = MockSystem(params)
    scenarios = [params]
    # Define the datamodule
    initial_domain = [
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    dm = EpisodicDataModule(system, initial_domain)
    experiment_suite = ExperimentSuite([])
    controller = NeuralCLBFController(system, scenarios, dm, experiment_suite)

    # Create the points (state and control) at which to test the Lie derivatives
    N_test = 2
    x = torch.Tensor(N_test, system.n_dims).uniform_(-1.0, 1.0)
    u = torch.Tensor(N_test, system.n_controls).uniform_(-1.0, 1.0)

    # Compute the Lie derivatives and expected change in V
    Lf_V, Lg_V = controller.V_lie_derivatives(x)
    Vdot = Lf_V + torch.bmm(Lg_V, u.unsqueeze(-1))

    # To validate the Lie derivatives, simulate V forward and approximate the derivative
    delta_t = 0.001
    V_now = controller.V(x)
    xdot = system.closed_loop_dynamics(x, u)
    x_next = x + xdot * delta_t
    V_next = controller.V(x_next)
    Vdot_simulated = (V_next - V_now) / delta_t

    # Make sure they're close
    assert torch.allclose(
        Vdot.squeeze(), Vdot_simulated.squeeze(), atol=0.001, rtol=1e-2
    )
