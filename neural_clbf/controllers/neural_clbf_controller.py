import itertools
from typing import Tuple, List, Optional, Callable, Union
from collections import OrderedDict
import random

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.utils import Controller
from neural_clbf.experiments.common.episodic_datamodule import EpisodicDataModule


class NeuralCLBFController(pl.LightningModule):
    """
    A neural rCLBF controller
    """

    controller_period: float

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        datamodule: EpisodicDataModule,
        clbf_hidden_layers: int = 2,
        clbf_hidden_size: int = 48,
        u_nn_hidden_layers: int = 1,
        u_nn_hidden_size: int = 8,
        clbf_lambda: float = 1.0,
        safety_level: float = 1.0,
        goal_level: float = 0.0,
        clbf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-3,
        epochs_per_episode: int = 5,
        penalty_scheduling_rate: float = 0.0,
        num_init_epochs: int = 5,
        plotting_callbacks: Optional[
            List[Callable[[Controller], Tuple[str, figure]]]
        ] = None,
        vary_safe_level: bool = False,
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
            safety_level: safety level set value for the CLBF
            goal_level: goal level set value for the CLBF
            clbf_relaxation_penalty: the penalty for relaxing CLBF conditions.
            controller_period: the timestep to use in simulating forward Vdot
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the CLBF decrease loss
            epochs_per_episode: the number of epochs to include in each episode
            penalty_scheduling_rate: the rate at which to ramp the rollout relaxation
                                     penalty up to clbf_relaxation_penalty. Set to 0 to
                                     disable penalty scheduling (use constant penalty)
            num_init_epochs: the number of epochs to pretrain the controller on the
                             linear controller
            plotting_callbacks: a list of plotting functions that each take a
                                NeuralCLBFController and return a tuple of a string
                                name and figure object to log
            vary_safe_level: if True, optimize the safe level as a parameter
        """
        super().__init__()
        self.save_hyperparameters()

        # Save the provided model
        self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the datamodule
        self.datamodule = datamodule

        # Save the other parameters
        self.clbf_lambda = clbf_lambda
        self.vary_safe_level = vary_safe_level
        self.safe_level: Union[torch.Tensor, float]
        self.unsafe_level: Union[torch.Tensor, float]
        if vary_safe_level:
            self.safe_level = nn.parameter.Parameter(torch.tensor(safety_level))
        else:
            self.safe_level = safety_level
        self.goal_level = goal_level
        self.unsafe_level = self.safe_level
        self.clbf_relaxation_penalty = clbf_relaxation_penalty
        self.controller_period = controller_period
        self.primal_learning_rate = primal_learning_rate
        self.epochs_per_episode = epochs_per_episode
        self.penalty_scheduling_rate = penalty_scheduling_rate
        self.num_init_epochs = num_init_epochs

        # Compute and save the center and range of the state variables
        x_max, x_min = dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_range = (x_max - x_min) / 2.0
        # Scale to get the input between (-k, k), centered at 0
        k = 1.0
        self.x_range = self.x_range / k
        # We shouldn't scale or offset any angle dimensions
        self.x_center[self.dynamics_model.angle_dims] = 0.0
        self.x_range[self.dynamics_model.angle_dims] = 1.0

        # Get plotting callbacks
        if plotting_callbacks is None:
            plotting_callbacks = []
        self.plotting_callbacks = plotting_callbacks

        # Some of the dimensions might represent angles. We want to replace these
        # dimensions with two dimensions: sin and cos of the angle. To do this, we need
        # to figure out how many numbers are in the expanded state
        n_angles = len(self.dynamics_model.angle_dims)
        self.n_dims_extended = self.dynamics_model.n_dims + n_angles

        # Define the CLBF network, which we denote V
        self.clbf_hidden_layers = clbf_hidden_layers
        self.clbf_hidden_size = clbf_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(
            self.n_dims_extended, self.clbf_hidden_size
        )
        self.V_layers["input_activation"] = nn.Tanh()
        for i in range(self.clbf_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.clbf_hidden_size, self.clbf_hidden_size
            )
            if i < self.clbf_hidden_layers - 1:
                self.V_layers[f"layer_{i}_activation"] = nn.Tanh()
        # self.V_layers["output_linear"] = nn.Linear(self.clbf_hidden_size, 1)
        self.V_nn = nn.Sequential(self.V_layers)

        # Also define the proof controller network, denoted u_nn
        self.u_nn_hidden_layers = u_nn_hidden_layers
        self.u_nn_hidden_size = u_nn_hidden_size
        # Likewise, build the network up layer by layer, starting with the input
        self.u_nn_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.u_nn_layers["input_linear"] = nn.Linear(
            self.n_dims_extended, self.u_nn_hidden_size
        )
        self.u_nn_layers["input_activation"] = nn.Tanh()
        for i in range(self.u_nn_hidden_layers):
            self.u_nn_layers[f"layer_{i}_linear"] = nn.Linear(
                self.u_nn_hidden_size, self.u_nn_hidden_size
            )
            self.u_nn_layers[f"layer_{i}_activation"] = nn.Tanh()
        # Tanh output activation, so the control saturates at [-1, 1]
        self.u_nn_layers["output_linear"] = nn.Linear(
            self.u_nn_hidden_size, self.dynamics_model.n_controls
        )
        self.u_nn_layers["output_activation"] = nn.Tanh()
        self.u_nn = nn.Sequential(self.u_nn_layers)

    def prepare_data(self):
        return self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        return self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input using the stored center point and range

        args:
            x: bs x self.dynamics_model.n_dims the points to normalize
        """
        return (x - self.x_center.type_as(x)) / self.x_range.type_as(x)

    def normalize_with_angles(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input using the stored center point and range, and replace all
        angles with the sine and cosine of the angles

        args:
            x: bs x self.dynamics_model.n_dims the points to normalize
        """
        # Scale and offset based on the center and range
        x = self.normalize(x)

        # Replace all angles with their sine, and append cosine
        angle_dims = self.dynamics_model.angle_dims
        angles = x[:, angle_dims]
        x[:, angle_dims] = torch.sin(angles)
        x = torch.cat((x, torch.cos(angles)), dim=-1)

        return x

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # Apply the offset and range to normalize about zero
        x = self.normalize_with_angles(x)

        # Compute the CLBF layer-by-layer, computing the Jacobian alongside

        # We need to initialize the Jacobian to reflect the normalization that's already
        # been done to x
        bs = x.shape[0]
        JV = torch.zeros(
            (bs, self.n_dims_extended, self.dynamics_model.n_dims)
        ).type_as(x)
        # and for each non-angle dimension, we need to scale by the normalization
        for dim in range(self.dynamics_model.n_dims):
            JV[:, dim, dim] = 1.0 / self.x_range[dim].type_as(x)

        # And adjust the Jacobian for the angle dimensions
        for offset, sin_idx in enumerate(self.dynamics_model.angle_dims):
            cos_idx = self.dynamics_model.n_dims + offset
            JV[:, sin_idx, sin_idx] = x[:, cos_idx]
            JV[:, cos_idx, sin_idx] = -x[:, sin_idx]

        # Now step through each layer in V
        V = x
        for layer in self.V_nn:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.Tanh):
                JV = torch.matmul(torch.diag_embed(1 - V ** 2), JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)

        # Compute the final activation
        JV = torch.bmm(V.unsqueeze(1), JV)
        V = 0.5 * (V * V).sum(dim=1) - self.goal_level

        # # Lol JK use lqr V
        # # Get the nominal Lyapunov function
        # P = self.dynamics_model.P.type_as(x)
        # # Reshape to use pytorch's bilinear function
        # P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        # V = 0.5 * F.bilinear(x, x, P)
        # P = P.reshape(self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        # JV = F.linear(x, P)
        # JV = JV.reshape(x.shape[0], 1, self.dynamics_model.n_dims)

        # # Make a gradient
        # x = self.normalize_with_angles(x)
        # V_net = self.V_nn(x)
        # V += 0.0000001 * (V_net * V_net).sum(dim=1).unsqueeze(-1) - self.goal_level

        return V, JV

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the value of the CLBF"""
        V, _ = self.V_with_jacobian(x)
        return V

    def u(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the learned controller input from the state x

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the
               controller
        """
        # Apply the offset and range to normalize about zero
        x_norm = self.normalize_with_angles(x)

        # Compute the control effort using the neural network
        u = self.u_nn(x_norm)

        # Scale to reflect plant actuator limits
        upper_lim, lower_lim = self.dynamics_model.control_limits
        u_center = (upper_lim + lower_lim).type_as(x_norm) / 2.0
        u_semi_range = (upper_lim - lower_lim).type_as(x_norm) / 2.0

        u_scaled = u * u_semi_range + u_center

        # # For now, set u to u_nominal to test V learning
        # # TODO @dawsonc, not permanent
        # u_scaled = self.dynamics_model.u_nominal(x) + 0.00001 * u_scaled

        return u_scaled

    def V_lie_derivatives(
        self, x: torch.Tensor, scenarios: Optional[ScenarioList] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Lie derivatives of the CLBF V along the control-affine dynamics

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            scenarios: optional list of scenarios. Defaults to self.scenarios
        returns:
            Lf_V: bs x len(scenarios) x 1 tensor of Lie derivatives of V
                  along f
            Lg_V: bs x len(scenarios) x self.dynamics_model.n_controls tensor
                  of Lie derivatives of V along g
        """
        if scenarios is None:
            scenarios = self.scenarios
        n_scenarios = len(scenarios)

        # Get the Jacobian of V for each entry in the batch
        _, gradV = self.V_with_jacobian(x)

        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, n_scenarios, 1)
        Lg_V = torch.zeros(batch_size, n_scenarios, self.dynamics_model.n_controls)
        Lf_V = Lf_V.type_as(x)
        Lg_V = Lg_V.type_as(x)

        for i in range(n_scenarios):
            # Get the dynamics f and g for this scenario
            s = scenarios[i]
            f, g = self.dynamics_model.control_affine_dynamics(x, params=s)

            # Multiply these with the Jacobian to get the Lie derivatives
            Lf_V[:, i, :] = torch.bmm(gradV, f).squeeze(1)
            Lg_V[:, i, :] = torch.bmm(gradV, g).squeeze(1)

        # return the Lie derivatives
        return Lf_V, Lg_V

    def solve_CLBF_QP(
        self, x, relaxation_penalty: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            relaxation_penalty: the penalty to use for CLBF relaxation, defaults to
                                self.clbf_relaxation_penalty
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLBF had to be relaxed in each
                        case
            objectives: bs x 1 tensor of the QP objective.
        """
        # Get the value of the CLBF and its Lie derivatives
        V = self.V(x)
        Lf_V, Lg_V = self.V_lie_derivatives(x)

        # Get the nn control input as well
        u_nn = self.u(x)

        # Apply default penalty if needed
        if relaxation_penalty is None:
            relaxation_penalty = self.clbf_relaxation_penalty

        # To find the control input, we want to solve a QP constrained by
        #
        # L_f V + L_g V u + lambda V <= 0
        #
        # To ensure that this QP is always feasible, we relax the constraint
        #
        # L_f V + L_g V u + lambda V - r <= 0
        #                              r >= 0
        #
        # and add the cost term relaxation_penalty * r.
        #
        # We want the objective to be to minimize
        #
        #           ||u - u_nn||^2 + relaxation_penalty * r^2
        #
        # This reduces to (ignoring constant terms)
        #
        #           u^T I u - 2 u_nn^T u + relaxation_penalty * r^2

        n_controls = self.dynamics_model.n_controls
        n_scenarios = self.n_scenarios
        allow_relaxation = relaxation_penalty < 1e6

        # Solve a QP for each row in x
        bs = x.shape[0]
        u_result = torch.zeros(bs, n_controls)
        r_result = torch.zeros(bs, n_scenarios)
        objective_result = torch.zeros(bs, 1)
        for batch_idx in range(bs):
            # Skip any bad points
            if (
                torch.isnan(x[batch_idx]).any()
                or torch.isinf(x[batch_idx]).any()
                or torch.isnan(Lg_V[batch_idx]).any()
                or torch.isinf(Lg_V[batch_idx]).any()
                or torch.isnan(Lf_V[batch_idx]).any()
                or torch.isinf(Lf_V[batch_idx]).any()
            ):
                continue

            # Instantiate the model
            model = gp.Model("clbf_qp")
            # Create variables for control input and (optionally) the relaxations
            upper_lim, lower_lim = self.dynamics_model.control_limits
            upper_lim = upper_lim.cpu().numpy()
            lower_lim = lower_lim.cpu().numpy()
            u = model.addMVar(n_controls, lb=lower_lim, ub=upper_lim)
            if allow_relaxation:
                r = model.addMVar(n_scenarios, lb=0, ub=GRB.INFINITY)

            # Define the cost
            Q = np.eye(n_controls)
            u_nn_np = u_nn[batch_idx, :].detach().cpu().numpy()
            objective = u @ Q @ u - 2 * u_nn_np @ Q @ u + u_nn_np @ Q @ u_nn_np
            if allow_relaxation:
                relax_penalties = relaxation_penalty * np.ones(n_scenarios)
                objective += relax_penalties @ r

            # Now build the CLBF constraints
            for i in range(n_scenarios):
                Lg_V_np = Lg_V[batch_idx, i, :].detach().cpu().numpy()
                Lf_V_np = Lf_V[batch_idx, i, :].detach().cpu().numpy()
                V_np = V[batch_idx].detach().cpu().numpy()
                clbf_constraint = Lf_V_np + Lg_V_np @ u + self.clbf_lambda * V_np
                if allow_relaxation:
                    clbf_constraint -= r[i]
                model.addConstr(clbf_constraint <= 0.0, name=f"Scenario {i} Decrease")

            # Optimize!
            model.setParam("DualReductions", 0)
            model.setObjective(objective, GRB.MINIMIZE)
            model.optimize()

            if model.status != GRB.OPTIMAL:
                # Make the relaxations nan if the problem was infeasible, as a signal
                # that something has gone wrong
                if allow_relaxation:
                    for i in range(n_scenarios):
                        r_result[batch_idx, i] = torch.tensor(float("nan"))
                continue

            # Extract the results
            for i in range(n_controls):
                u_result[batch_idx, i] = torch.tensor(u[i].x)
            if allow_relaxation:
                for i in range(n_scenarios):
                    r_result[batch_idx, i] = torch.tensor(r[i].x)
            objective_result[batch_idx, 0] = torch.tensor(model.objVal)

        return u_result.type_as(x), r_result.type_as(x), objective_result.type_as(x)

    def forward(self, x):
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        u, _, _ = self.solve_CLBF_QP(x)
        return u

    def boundary_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        dist_to_goal: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to boundary conditions

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            dist_to_goal: the distance from x to the goal region
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        eps = 1e-2
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        V = self.V(x)

        #   1.) CLBF should be minimized on the goal point
        V_goal_pt = self.V(self.dynamics_model.goal_point.type_as(x))
        goal_term = V_goal_pt.mean()
        loss.append(("CLBF goal term", goal_term))

        #   2.) 0 < V <= safe_level in the safe region
        V_safe = V[safe_mask]
        safe_violation = F.relu(eps + V_safe - self.safe_level)
        safe_V_term = 1e2 * safe_violation.mean()
        loss.append(("CLBF safe region term", safe_V_term))
        if accuracy:
            safe_V_acc = (safe_violation <= eps).sum() / safe_violation.nelement()
            loss.append(("CLBF safe region accuracy", safe_V_acc))

        #   3.) V >= unsafe_level in the unsafe region
        V_unsafe = V[unsafe_mask]
        unsafe_violation = F.relu(eps + self.unsafe_level - V_unsafe)
        unsafe_V_term = 1e2 * unsafe_violation.mean()
        loss.append(("CLBF unsafe region term", unsafe_V_term))
        if accuracy:
            unsafe_V_acc = (unsafe_violation <= eps).sum() / unsafe_violation.nelement()
            loss.append(("CLBF unsafe region accuracy", unsafe_V_acc))

        # #   4.) V >= eps * ||x||^2
        # eps = 0.1
        # lower_bound = F.relu(eps * (x ** 2).sum(dim=-1) - V)
        # lower_bound_term = lower_bound.mean()
        # loss.append(("Lower bound term", lower_bound_term))
        # if accuracy:
        #     unsafe_V_acc = (lower_bound <= eps).sum() / lower_bound.nelement()
        #     loss.append(("Lower bound term", unsafe_V_acc))

        return loss

    def descent_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        dist_to_goal: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to the descent condition

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            dist_to_goal: the distance from x to the goal region
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        #   1.) A term to encourage satisfaction of the CLBF decrease condition,
        # which requires that V is decreasing everywhere where V <= safe_level
        V = self.V(x)

        # First figure out where this condition needs to hold
        condition_active = V < self.safe_level

        # Now compute the decrease in that region, using the proof controller
        clbf_descent_term_lin = torch.tensor(0.0).type_as(x)
        clbf_descent_acc_lin = torch.tensor(0.0).type_as(x)
        # Get the current value of the CLBF and its Lie derivatives
        # (Lie derivatives are computed using a linear fit of the dynamics)
        # TODO @dawsonc do we need dynamics learning here?
        Lf_V, Lg_V = self.V_lie_derivatives(x)
        # Get the control and reshape it to bs x n_controls x 1
        u_nn = self.u(x)
        eps = 0.0
        for i, s in enumerate(self.scenarios):
            # Use the dynamics to compute the derivative of V
            Vdot = Lf_V[:, i, :].unsqueeze(1) + torch.bmm(
                Lg_V[:, i, :].unsqueeze(1),
                u_nn.reshape(-1, self.dynamics_model.n_controls, 1),
            )
            Vdot = Vdot.reshape(V.shape)
            violation = F.relu(eps + Vdot + self.clbf_lambda * V)
            violation = violation[condition_active]
            clbf_descent_term_lin += violation.mean()
            clbf_descent_acc_lin += (violation <= eps).sum() / (
                violation.nelement() * self.n_scenarios
            )

        loss.append(("CLBF descent term (linearized)", clbf_descent_term_lin))
        if accuracy:
            loss.append(("CLBF descent accuracy (linearized)", clbf_descent_acc_lin))

        #   2.) A term to encourage satisfaction of CLF condition, using the method from
        # the RSS paper.
        # We compute the change in V in two ways: simulating x forward in time and check
        # if V decreases in each scenario
        eps = 0.0
        clbf_descent_term_sim = torch.tensor(0.0).type_as(x)
        clbf_descent_acc_sim = torch.tensor(0.0).type_as(x)
        for s in self.scenarios:
            xdot = self.dynamics_model.closed_loop_dynamics(x, u_nn, params=s)
            x_next = x + self.dynamics_model.dt * xdot
            V_next = self.V(x_next)
            violation = F.relu(
                eps + (V_next - V) / self.controller_period + self.clbf_lambda * V
            )
            violation = violation[condition_active]

            clbf_descent_term_sim += violation.mean()
            clbf_descent_acc_sim += (violation <= eps).sum() / (
                violation.nelement() * self.n_scenarios
            )
        loss.append(("CLBF descent term (simulated)", clbf_descent_term_sim))
        if accuracy:
            loss.append(("CLBF descent accuracy (simulated)", clbf_descent_acc_sim))

        return loss

    def initial_V_loss(self, x: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        """
        Compute the loss during the initialization epochs, which trains the net to
        match the local linear lyapunov function
        """
        loss = []

        # The initial losses should decrease exponentially to zero, based on the epoch
        epoch_count = max(self.current_epoch - self.num_init_epochs, 0)
        decrease_factor = 0.5 ** epoch_count

        #   1.) Compare the CLBF to the nominal solution
        # Get the learned CLBF
        V = self.V(x)

        # Get the nominal Lyapunov function
        P = self.dynamics_model.P.type_as(x)
        # Reshape to use pytorch's bilinear function
        P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        V_nominal = 0.5 * F.bilinear(x, x, P).squeeze()

        # Compute the error between the two
        clbf_mse_loss = (V - V_nominal) ** 2
        clbf_mse_loss = decrease_factor * clbf_mse_loss.mean()
        loss.append(("CLBF MSE", clbf_mse_loss))

        # #   2.) Ensure that V >= 0.1 * nominal solution
        # clbf_lower_bound_loss = 10 * F.relu(0.1 * V_nominal - V).mean()
        # loss.append(("CLBF Bound", clbf_lower_bound_loss))

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask, dist_to_goal = batch

        # Compute the losses
        component_losses = {}
        if self.opt_idx_dict[optimizer_idx] == "clbf":
            component_losses.update(self.initial_V_loss(x))
            if self.current_epoch > self.num_init_epochs:
                component_losses.update(
                    self.boundary_loss(
                        x, goal_mask, safe_mask, unsafe_mask, dist_to_goal
                    )
                )
                component_losses.update(
                    self.descent_loss(
                        x, goal_mask, safe_mask, unsafe_mask, dist_to_goal
                    )
                )
        else:
            # component_losses.update(
            #     self.descent_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
            # )
            u_nominal = self.dynamics_model.u_nominal(x)
            u_nn = self.u(x)
            component_losses.update(
                {"Controller MSE": ((u_nn - u_nominal) ** 2).sum(dim=-1).mean()}
            )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        batch_dict = {"loss": total_loss, **component_losses}

        return batch_dict

    def training_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Outputs contains a list for each optimizer, and we need to collect the losses
        # from all of them if there is a nested list
        if isinstance(outputs[0], list):
            outputs = itertools.chain(*outputs)

        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / train", avg_losses["loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "loss":
                continue
            # Log the other losses
            self.log(loss_key + " / train", avg_losses[loss_key], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask, dist_to_goal = batch

        # Get the various losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
        )
        component_losses.update(
            self.descent_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
        )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        # Also compute the accuracy associated with each loss
        component_losses.update(
            self.boundary_loss(
                x, goal_mask, safe_mask, unsafe_mask, dist_to_goal, accuracy=True
            )
        )
        component_losses.update(
            self.descent_loss(
                x, goal_mask, safe_mask, unsafe_mask, dist_to_goal, accuracy=True
            )
        )

        batch_dict = {"val_loss": total_loss, **component_losses}

        return batch_dict

    def validation_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / val", avg_losses["val_loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "val_loss":
                continue
            # Log the other losses
            self.log(loss_key + " / val", avg_losses[loss_key], sync_dist=True)

        # **Now entering spicetacular automation zone**
        # We automatically plot and save the CLBF and some simulated rollouts
        # at the end of the validation epoch, using arbitrary plotting callbacks!

        # Only plot every 5 epochs
        if self.current_epoch % 5 != 0:
            return

        # Figure out the relaxation penalty for this rollout
        if self.penalty_scheduling_rate > 0:
            relaxation_penalty = (
                self.clbf_relaxation_penalty
                * self.current_epoch
                / self.penalty_scheduling_rate
            )
        else:
            relaxation_penalty = self.clbf_relaxation_penalty
        old_relaxation_penalty = self.clbf_relaxation_penalty
        self.clbf_relaxation_penalty = relaxation_penalty

        plots = []
        for plot_fn in self.plotting_callbacks:
            plot_name, plot = plot_fn(self)
            self.logger.experiment.add_figure(
                plot_name, plot, global_step=self.current_epoch
            )
            plots.append(plot)
        self.logger.experiment.close()
        self.logger.experiment.flush()
        [plt.close(plot) for plot in plots]

        # Restore the nominal relaxation penalty in case any plotting callbacks
        # do parameter sweeps
        self.clbf_relaxation_penalty = old_relaxation_penalty

    @pl.core.decorators.auto_move_data
    def simulator_fn(
        self,
        x_init: torch.Tensor,
        num_steps: int,
        use_qp: bool = True,
        relaxation_penalty: Optional[float] = None,
    ):
        if use_qp:

            def controller_fn(x):
                u, _, _ = self.solve_CLBF_QP(x, relaxation_penalty)
                return u

        else:

            def controller_fn(x):
                return self.u(x)

        # Choose parameters randomly
        random_scenario = {}
        for param_name in self.scenarios[0].keys():
            param_max = max([s[param_name] for s in self.scenarios])
            param_min = min([s[param_name] for s in self.scenarios])
            random_scenario[param_name] = random.uniform(param_min, param_max)

        return self.dynamics_model.simulate(
            x_init,
            num_steps,
            controller_fn,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.controller_period,
            params=random_scenario,
        )

    def on_validation_epoch_end(self):
        """This function is called at the end of every validation epoch"""
        # We want to generate new data at the end of every episode
        if self.current_epoch > 0 and self.current_epoch % self.epochs_per_episode == 0:
            if self.penalty_scheduling_rate > 0:
                relaxation_penalty = (
                    self.clbf_relaxation_penalty
                    * self.current_epoch
                    / self.penalty_scheduling_rate
                )
            else:
                relaxation_penalty = self.clbf_relaxation_penalty

            # Use the models simulation function with this controller
            def simulator_fn_wrapper(x_init: torch.Tensor, num_steps: int):
                return self.simulator_fn(
                    x_init,
                    num_steps,
                    use_qp=True,
                    relaxation_penalty=relaxation_penalty,
                )

            self.datamodule.add_data(simulator_fn_wrapper)

    def configure_optimizers(self):
        clbf_params = list(self.V_nn.parameters())
        if self.vary_safe_level:
            clbf_params += [self.safe_level]

        clbf_opt = torch.optim.SGD(
            clbf_params,
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )
        u_opt = torch.optim.SGD(
            self.u_nn.parameters(),
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "clbf", 1: "controller"}

        return [clbf_opt, u_opt]

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
        optimizer.zero_grad()

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # During initialization epochs, step both
        if epoch <= self.num_init_epochs:
            optimizer.step(closure=optimizer_closure)
            return

        # Otherwise, switch between the controller and CLBF every few epochs
        if self.opt_idx_dict[optimizer_idx] == "clbf":
            if (epoch - self.num_init_epochs) % 40 < 20:
                optimizer.step(closure=optimizer_closure)

        if self.opt_idx_dict[optimizer_idx] == "controller":
            if (epoch - self.num_init_epochs) % 40 >= 20:
                optimizer.step(closure=optimizer_closure)
