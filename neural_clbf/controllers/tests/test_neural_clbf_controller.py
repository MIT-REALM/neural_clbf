"""Test the the vanilla neural clbf controller"""
import torch
import random

from torch.autograd.functional import jacobian

from neural_clbf.controllers.neural_clbf_controller import (
    NeuralCLBFController,
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
    controller = NeuralCLBFController(system, scenarios, dm)
    assert controller is not None


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
    controller = NeuralCLBFController(system, scenarios, dm)

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
    controller = NeuralCLBFController(system, scenarios, dm)

    # Create the points (state and control) at which to test the Lie derivatives
    N_test = 2
    x = torch.Tensor(N_test, system.n_dims).uniform_(-1.0, 1.0)
    u = torch.Tensor(N_test, system.n_controls).uniform_(-1.0, 1.0)

    # Compute the Lie derivatives and expected change in V
    Lf_V, Lg_V = controller.V_lie_derivatives(x)
    Vdot = Lf_V + torch.bmm(Lg_V, u.unsqueeze(-1))

    # To validate the Lie derivatives, simulate V forward and approximate the derivative
    delta_t = 0.0001
    V_now = controller.V(x)
    xdot = system.closed_loop_dynamics(x, u)
    x_next = x + xdot * delta_t
    V_next = controller.V(x_next)
    Vdot_simulated = (V_next - V_now) / delta_t

    # Make sure they're close
    assert torch.allclose(
        Vdot.squeeze(), Vdot_simulated.squeeze(), atol=0.001, rtol=1e-2
    )
