"""Test the data generation for the 2D quadrotor with obstacles"""
import torch
import random

from neural_clbf.controllers.neural_qp_rclbf_controller import (
    NeuralQPrCLBFController,
)
from neural_clbf.systems.tests.mock_system import MockSystem


def test_init_neuralqprclbfcontroller():
    """Test the initialization of a NeuralQPrCLBFController"""
    # Define the model system
    params = {}
    system = MockSystem(params)

    # Instantiate with a list of only one scenarios
    scenarios = [params]
    controller = NeuralQPrCLBFController(system, scenarios)
    assert controller is not None

    # Make sure we can get a control signal
    N_test = 10
    x = torch.Tensor(N_test, system.n_dims).uniform_(-1.0, 1.0)
    u = controller(x)
    assert u is not None
    assert u.shape[0] == N_test
    assert u.shape[1] == system.n_controls


def test_V_lie_derivatives():
    """Test computation of Lie Derivatives"""
    # Set a random seed for repeatability
    random.seed(0)

    # Create the controller
    params = {}
    system = MockSystem(params)
    scenarios = [params]
    controller = NeuralQPrCLBFController(system, scenarios)

    # Create the points (state and control) at which to test the Lie derivatives
    N_test = 10
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
