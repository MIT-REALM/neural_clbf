"""Test the the vanilla neural clbf controller"""
import torch
import random

from torch.autograd.functional import jacobian

from neural_clbf.controllers.neural_cbf_controller import (
    NeuralCBFController,
)
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.systems.tests.mock_system import MockSystem
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule


def test_init_neuralcbfcontroller():
    """Test the initialization of a NeuralCBFController"""
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
    controller = NeuralCBFController(system, scenarios, dm, experiment_suite)
    assert controller is not None
    assert controller.controller_period > 0


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
    controller = NeuralCBFController(
        system,
        scenarios,
        dm,
        experiment_suite,
        cbf_hidden_layers=2,
        cbf_hidden_size=64,
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
        deltaV_expected.squeeze(), deltaV_simulated.squeeze(), rtol=1e-2, atol=1e-4
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
    controller = NeuralCBFController(system, scenarios, dm, experiment_suite)

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
