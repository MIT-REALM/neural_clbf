"""Test the the vanilla neural clbf controller"""
import torch
import random

from torch.autograd.functional import jacobian

from neural_clbf.controllers.neural_bb_lbf_controller import (
    NeuralBlackBoxLBFController,
)
from neural_clbf.systems.tests.mock_system import MockSystem
from neural_clbf.experiments.common.episodic_datamodule import EpisodicDataModule


def test_init_neuralrclbfcontroller():
    """Test the initialization of a NeuralBlackBoxLBFController"""
    # Define the model system
    params = {}
    system = MockSystem(params)
    # Define the datamodule
    initial_domain = [
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    dm = EpisodicDataModule(system, initial_domain)

    # Instantiate the controller
    controller = NeuralBlackBoxLBFController(system, dm)
    assert controller is not None

    # Make sure the neural nets are the right size
    n_dims_extended = system.n_dims + len(system.angle_dims)
    assert controller.V_nn[0].in_features == n_dims_extended
    assert controller.V_nn[-1].out_features == 1
    assert controller.u_nn[0].in_features == n_dims_extended
    assert controller.u_nn[-2].out_features == system.n_controls
    assert controller.f_nn[0].in_features == n_dims_extended
    assert controller.f_nn[-1].out_features == system.n_dims
    assert controller.g_nn[0].in_features == n_dims_extended
    assert controller.g_nn[-1].out_features == system.n_dims * system.n_controls


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

    # Instantiate the controller
    controller = NeuralBlackBoxLBFController(system, dm)

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
    # Define the datamodule
    initial_domain = [
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    dm = EpisodicDataModule(system, initial_domain)
    controller = NeuralBlackBoxLBFController(system, dm)

    # Create the points and perturbations with which to test the Jacobian
    N_test = 10
    x = torch.Tensor(N_test, system.n_dims).uniform_(-1.0, 1.0)
    dx = torch.Tensor(N_test, system.n_dims).uniform_(1e-2, 2e-2)

    # Compute V and its Jacobian
    V, gradV = controller.V_with_jacobian(x)
    # and use these to get the expected change
    deltaV_expected = torch.bmm(gradV, dx.unsqueeze(-1))

    # Compare first with the autodiff Jacobian
    J_V_x = torch.zeros(N_test, 1, x.shape[1])
    for i in range(N_test):
        J_V_x[i, :, :] = jacobian(controller.V, x[i, :].unsqueeze(0))

    # To validate the Jacobian, approximate with finite difference
    x_next = x + dx
    V_next = controller.V(x_next)
    deltaV_simulated = V_next - V

    # Make sure they're close
    tol = 0.1 * deltaV_simulated.abs().mean().item()
    assert torch.allclose(
        deltaV_expected.squeeze(), deltaV_simulated.squeeze(), atol=tol
    )


def test_V_dot():
    """Test computation of Lie Derivatives"""
    # Set a random seed for repeatability
    random.seed(0)
    torch.manual_seed(0)

    # Create the controller
    params = {}
    system = MockSystem(params)
    # Define the datamodule
    initial_domain = [
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    dm = EpisodicDataModule(system, initial_domain)
    controller = NeuralBlackBoxLBFController(system, dm)

    # Create the points (state and control) at which to test the Lie derivatives
    N_test = 2
    x = torch.Tensor(N_test, system.n_dims).uniform_(-1.0, 1.0)
    u = torch.Tensor(N_test, system.n_controls).uniform_(-1.0, 1.0)

    # Compute the Lie derivatives and expected change in V
    _, Vdot = controller.Vdot(x, u)

    # To validate the Lie derivatives, simulate V forward using the *learned* dynamics
    # Since we don't do any training here, we need to use the learned dynamics to be
    # consistent
    delta_t = 0.0001
    V_now = controller.V(x)
    f, g = controller.learned_dynamics(x)
    xdot = f + torch.bmm(g, u.unsqueeze(-1)).squeeze()
    x_next = x + xdot * delta_t
    V_next = controller.V(x_next)
    Vdot_simulated = (V_next - V_now) / delta_t

    # Make sure they're close
    assert torch.allclose(
        Vdot.squeeze(), Vdot_simulated.squeeze(), atol=0.001, rtol=1e-2
    )
