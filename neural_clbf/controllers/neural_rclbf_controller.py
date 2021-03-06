from typing import (
    Tuple,
)
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
import pytorch_lightning as pl

import casadi

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList


class NeuralrCLBFController(pl.LightningModule):
    """
    A neural rCLBF controller
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        clbf_hidden_layers: int = 2,
        clbf_hidden_size: int = 48,
        u_nn_hidden_layers: int = 3,
        u_nn_hidden_size: int = 48,
        clbf_lambda: float = 1.0,
        clbf_safety_level: float = 1.0,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            clbf_hidden_layers: number of hidden layers to use for the CLBF network
            clbf_hidden_size: number of neurons per hidden layer in the CLBF network
            u_nn_hidden_layers: number of hidden layers to use for the proof controller
            u_nn_hidden_size: number of neurons per hidden layer in the proof controller
            clbf_lambda: convergence rate for the CLBF
            clbf_safety_level: safety level set value for the CLBF
        """
        super().__init__()

        # Save the provided model
        self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Define the CLBF network, which we denote V
        self.clbf_hidden_layers = clbf_hidden_layers
        self.clbf_hidden_size = clbf_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_layer"] = nn.Linear(
            self.dynamics_model.n_dims, self.clbf_hidden_size
        )
        self.V_layers["input_layer_activation"] = nn.Tanh()
        for i in range(self.clbf_hidden_layers):
            self.V_layers[f"layer_{i}"] = nn.Linear(
                self.clbf_hidden_size, self.clbf_hidden_size
            )
            self.V_layers[f"layer_{i}_activation"] = nn.Tanh()
        # TODO: currently experimenting with a network that does not have to be PSD
        # (as V = V_hidden.T * V_hidden would be). Keep in mind that this might change
        self.V_layers["output_layer"] = nn.Linear(self.clbf_hidden_size, 1)
        self.V = nn.Sequential(self.V_layers)

        # Also define the proof controller network, denoted u_nn
        self.clbf_hidden_layers = clbf_hidden_layers
        self.clbf_hidden_size = clbf_hidden_size
        # Likewise, build the network up layer by layer, starting with the input
        self.u_NN_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.u_NN_layers["input_layer"] = nn.Linear(
            self.dynamics_model.n_dims, self.clbf_hidden_size
        )
        self.u_NN_layers["input_layer_activation"] = nn.Tanh()
        for i in range(self.clbf_hidden_layers):
            self.u_NN_layers[f"layer_{i}"] = nn.Linear(
                self.clbf_hidden_size, self.clbf_hidden_size
            )
            self.u_NN_layers[f"layer_{i}_activation"] = nn.Tanh()
        # Finally, add the output layer
        self.u_NN_layers["output_layer"] = nn.Linear(
            self.clbf_hidden_size, self.dynamics_model.n_controls
        )
        self.u_NN = nn.Sequential(self.u_NN_layers)

    def V_lie_derivatives(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Lie derivatives of the CLBF V along the control-affine dynamics

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            Lf_V: bs x self.n_scenarios x 1 tensor of Lie derivatives of V
                  along f
            Lf_V: bs x self.n_scenarios x self.dynamics_model.n_controls tensor
                  of Lie derivatives of V along g
        """
        # Get the Jacobian of V for each entry in the batch
        batch_size = x.shape[0]
        J_V_x = torch.zeros(batch_size, 1, x.shape[1])
        # Since this might be called in a no_grad environment, we use the
        # enable_grad environment to temporarily accumulate gradients
        with torch.enable_grad():
            for i in range(batch_size):
                J_V_x[i, :, :] = jacobian(self.V, x[i, :], create_graph=True)

        # We need to compute Lie derivatives for each scenario
        Lf_V = torch.zeros(batch_size, self.n_scenarios, 1)
        Lg_V = torch.zeros(batch_size, self.n_scenarios, self.dynamics_model.n_controls)

        for i in range(self.n_scenarios):
            # Next, we need to dynamics f and g
            f, g = self.dynamics_model.control_affine_dynamics(x)

            # Multiply these with the Jacobian to get the Lie derivatives
            Lf_V[:, i, :] = torch.bmm(J_V_x, f).squeeze(1)
            Lg_V[:, i, :] = torch.bmm(J_V_x, g).squeeze(1)

        # return the Lie derivatives
        return Lf_V, Lg_V

    def forward(self, x):
        """Determine the control input for a given state by solving a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u_rclbf: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        # Get the value of the CLBF...
        V = self.V(x)
        # and the Lie derivatives of the CLBF for each scenario
        Lf_V, Lg_V = self.V_lie_derivatives(x)

        # To find a control input, we need to solve an optimization problem.
        # TODO @dawsonc review whether torch-native solvers are as good as casadi
        # (cvxpylayers was pretty bad in terms of accuracy, but maybe qpth is better).
        batch_size = x.shape[0]
        u_batched = torch.zeros(batch_size, self.dynamics_model.n_controls)
        for i in range(batch_size):
            # Create an optimization problem to find a good input
            opti = casadi.Opti()
            # The decision variables will be the control inputs
            u = opti.variable(self.dynamics_model.n_controls)

            # The objective is simple: minimize the size of the control input
            opti.minimize(casadi.sumsqr(u))

            # Add a constraint for CLBF decrease in each scenario
            for j in range(self.n_scenarios):
                opti.subject_to(
                    Lf_V[i, j, :] + Lg_V[i, j, :] @ u + self.clbf_lambda * V <= 0.0
                )

            # Set up the solver
            p_opts = {"expand": True, "print_time": 0}
            s_opts = {"max_iter": 1000, "print_level": 0, "sb": "yes"}
            opti.solver("ipopt", p_opts, s_opts)

            # Solve the QP
            solution = opti.solve()

            # Save the solution
            u_batched[i, :] = solution.value(u)

        return u_batched

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3, weight_decay=1e-6)
