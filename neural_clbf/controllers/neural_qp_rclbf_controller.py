from typing import Tuple, Dict, List, Optional, Callable
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
import pytorch_lightning as pl
from matplotlib.pyplot import figure

from qpth.qp import QPFunction

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.utils import Controller


class NeuralQPrCLBFController(pl.LightningModule):
    """
    A neural rCLBF controller
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        clbf_hidden_layers: int = 1,
        clbf_hidden_size: int = 8,
        clbf_lambda: float = 1.0,
        safety_level: float = 1.0,
        clbf_timestep: float = 0.01,
        control_loss_weight: float = 1e-6,
        qp_clbf_relaxation_penalty: float = 1e4,
        learning_rate: float = 1e-3,
        x_center: Optional[torch.Tensor] = None,
        x_range: Optional[torch.Tensor] = None,
        plotting_callbacks: Optional[
            List[Callable[[Controller], Tuple[str, figure]]]
        ] = None,
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
            clbf_timestep: the timestep to use in simulating forward Vdot
            control_loss_weight: the weight to apply to the control loss
            qp_clbf_relaxation_penalty: the penalty coefficient applied to the
                                        relaxation of the CLBF decrease conditions in
                                        the QP controller.
            learning_rate: the learning rate for SGD
            x_center: a dynamics_model.n_dims length tensor representing the center
                      point of the data
            x_range: a dynamics_model.n_dims length tensor representing the range of the
                     data
            plotting_callbacks: a list of plotting functions that each take a
                                NeuralrCLBFController and return a tuple of a string
                                name and figure object to log
        """
        super().__init__()

        # Save the provided model
        self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the other parameters
        self.clbf_lambda = clbf_lambda
        self.safety_level = safety_level
        self.clbf_timestep = clbf_timestep
        self.control_loss_weight = control_loss_weight
        self.qp_clbf_relaxation_penalty = qp_clbf_relaxation_penalty
        self.learning_rate = learning_rate

        # Save the center and range if provided
        if x_center is not None:
            self.x_center = x_center
        else:
            self.x_center = torch.zeros(self.dynamics_model.n_dims)

        if x_range is not None:
            self.x_range = x_range
        else:
            self.x_range = torch.ones(self.dynamics_model.n_dims)

        # Get plotting callbacks
        if plotting_callbacks is None:
            plotting_callbacks = []
        self.plotting_callbacks = plotting_callbacks

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
        self.V_layers["output_layer"] = nn.Linear(
            self.clbf_hidden_size, self.clbf_hidden_size
        )
        self.V_net = nn.Sequential(self.V_layers)
        # We also want to be able to add a bias to V as needed
        self.V_bias = nn.Linear(1, 1)

        # Also set up the objective and actuation limit constraints for the qp
        # controller (enforced as G_u @ u <= h)
        self.G_u = torch.zeros(
            (2 * self.dynamics_model.n_controls, self.dynamics_model.n_controls)
        )
        self.h_u = torch.zeros((2 * self.dynamics_model.n_controls, 1))
        upper_lim, lower_lim = self.dynamics_model.control_limits
        for j in range(self.dynamics_model.n_controls):
            # Upper limit u[j] <= upper_lim[j]
            self.G_u[2 * j, j] = 1.0
            self.h_u[2 * j, 0] = upper_lim[j]
            # Upper limit u[j] >= lower_lim[j] (-> -u[j] <= -lower_lim[j])
            self.G_u[2 * j + 1, j] = -1.0
            self.h_u[2 * j + 1, 0] = -lower_lim[j]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input using the stored center point and range

        args:
            x: bs x self.dynamics_model.n_dims the points to normalize
        """
        return (x - self.x_center.type_as(x)) / self.x_range.type_as(x)

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the CLBF value from the output layer of V_net

        args:
            x: bs x sellf.dynamics_model.n_dims the points at which to evaluate the CLBF
        """
        # Apply the offset and range to normalize about zero
        x = self.normalize(x)

        # Compute the CLBF as the sum-squares of the output layer activations
        V_output = self.V_net(x)
        V = 0.5 * (V_output * V_output).sum(-1).reshape(x.shape[0], 1)
        # and add the bias
        V = self.V_bias(V)

        return V

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
        J_V_x = J_V_x.type_as(x)
        # Since this might be called in a no_grad environment, we use the
        # enable_grad environment to temporarily accumulate gradients
        with torch.enable_grad():
            for i in range(batch_size):
                J_V_x[i, :, :] = jacobian(
                    self.V, x[i, :].unsqueeze(0), create_graph=True
                )

        # We need to compute Lie derivatives for each scenario
        Lf_V = torch.zeros(batch_size, self.n_scenarios, 1)
        Lg_V = torch.zeros(batch_size, self.n_scenarios, self.dynamics_model.n_controls)
        Lf_V = Lf_V.type_as(x)
        Lg_V = Lg_V.type_as(x)

        for i in range(self.n_scenarios):
            # Get the dynamics f and g for this scenario
            s = self.scenarios[i]
            f, g = self.dynamics_model.control_affine_dynamics(x, params=s)

            # Multiply these with the Jacobian to get the Lie derivatives
            Lf_V[:, i, :] = torch.bmm(J_V_x, f).squeeze(1)
            Lg_V[:, i, :] = torch.bmm(J_V_x, g).squeeze(1)

        # return the Lie derivatives
        return Lf_V, Lg_V

    def u(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the controller input from the state x

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the
               controller
        returns:
            u: bs x self.dynamics_model.n_controls the control input
            relaxation: bs x 1 the relaxation of the CLBF constraint in the QP
        """
        # Get the value of the CLBF...
        V = self.V(x)
        # and the Lie derivatives of the CLBF for each scenario
        Lf_V, Lg_V = self.V_lie_derivatives(x)

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
        # The decision variables here are z=[u r], so our quadratic cost is
        # 1/2 z^T Q z + p^T z. We want this cost to equal
        #
        #           ||u - u_nominal||^2 + relaxation_penalty * (r^2 + r)
        #
        # This reduces to (ignoring constant terms)
        #
        #           u^T I u - 2 u_nominal^T u + relaxation_penalty * r
        #
        # so we need
        #
        #           Q = [I 0
        #                0 relaxation_penalty]
        #           p = [-2 u_nominal^T relaxation_penalty]
        #
        # Expressing the constraints formally:
        #
        #       Gz <= h
        #
        # where h = [-L_f V - lambda V, 0]^T and G = [L_g V, -1
        #                                             ... repeated for each scenario
        #                                             0,     -1
        #                                             G_u,    0]
        # We also add the user-specified inequality constraints
        n_controls = self.dynamics_model.n_controls
        n_scenarios = self.n_scenarios
        bs = x.shape[0]

        # Start by building the cost
        Q = torch.zeros(bs, n_controls + 1, n_controls + 1).type_as(x)
        for j in range(n_controls):
            Q[:, j, j] = 1.0
        Q[:, -1, -1] = self.qp_clbf_relaxation_penalty + 0.01
        p = torch.zeros(bs, n_controls + 1).type_as(x)
        p[:, :-1] = -2.0 * self.dynamics_model.u_nominal(x)
        p[:, -1] = self.qp_clbf_relaxation_penalty

        # Now build the inequality constraints G @ [u r]^T <= h
        G = torch.zeros(
            bs, n_scenarios + 1 + self.G_u.shape[0], n_controls + 1
        ).type_as(x)
        h = torch.zeros(bs, n_scenarios + 1 + self.h_u.shape[0], 1).type_as(x)
        # CLBF decrease condition in each scenario
        for i in range(n_scenarios):
            G[:, i, :n_controls] = Lg_V[:, i, :]
            G[:, i, -1] = -1
            h[:, i, :] = -Lf_V[:, i, :] - self.clbf_lambda * V
        # Positive relaxation
        G[:, n_scenarios, -1] = -1
        h[:, n_scenarios, 0] = 0.0
        # Actuation limits
        G[:, n_scenarios + 1 :, :n_controls] = self.G_u.type_as(x)
        h[:, n_scenarios + 1 :, 0] = self.h_u.view(1, -1).type_as(x)
        h = h.squeeze()
        # No equality constraints
        A = torch.tensor([])
        b = torch.tensor([])

        # Convert to double precision for solving
        Q = Q.double()
        p = p.double()
        G = G.double()
        h = h.double()

        # Solve the QP!
        result: torch.Tensor = QPFunction(verbose=False)(Q, p, G, h, A, b)
        # Extract the results
        u = result[:, :n_controls]
        relaxation = result[:, -1]

        return u, relaxation

    def forward(self, x):
        """Determine the control input for a given state by solving a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        # We don't need to return the relaxation on the forward pass
        u, _ = self.u(x)

        return u

    def clbf_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute a loss to train the CLBF

        args:
            x: the points at which to evaluate the loss
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
        returns:
            loss: a dictionary containing the losses in each category
        """
        eps = 1e-2
        # Compute loss to encourage satisfaction of the following conditions...
        loss = {}
        #   1.) CLBF value should be non-positive on the goal set.
        V0 = self.V(x[goal_mask])
        goal_term = F.relu(eps + V0)
        loss["CLBF goal term"] = goal_term.mean()

        #   2.) 0 <= V <= safe_level in the safe region (ignoring the goal)
        safe_mask = torch.logical_and(safe_mask, torch.logical_not(goal_mask))
        V_safe = self.V(x[safe_mask])
        safe_clbf_term = F.relu(eps + V_safe - self.safety_level) + F.relu(eps - V_safe)
        loss["CLBF safe region term"] = safe_clbf_term.mean()

        #   3.) V >= safe_level in the unsafe region
        V_unsafe = self.V(x[unsafe_mask])
        unsafe_clbf_term = F.relu(eps + self.safety_level - V_unsafe)
        loss["CLBF unsafe region term"] = unsafe_clbf_term.mean()

        return loss

    def controller_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute a loss to train the controller

        args:
            x: the points at which to evaluate the loss
        returns:
            loss: a dictionary containing the losses in each category
        """
        loss = {}

        #   Begin with a term to encourage satisfaction of CLBF decrease condition
        # We compute the change in V by simulating x forward in time and checking if V
        # decreases in each scenario
        clbf_descent_term_sim = torch.tensor(0.0).type_as(x)
        V = self.V(x)
        u, relaxation = self.u(x)
        eps = 0.01
        for s in self.scenarios:
            xdot = self.dynamics_model.closed_loop_dynamics(x, u, s)
            x_next = x + self.clbf_timestep * xdot
            V_next = self.V(x_next)
            clbf_descent_term_sim += F.relu(
                eps + V_next - (1 - self.clbf_lambda * self.clbf_timestep) * V
            ).mean()
        loss["CLBF descent term (simulated)"] = clbf_descent_term_sim

        # Also penalize the relaxation
        relaxation_term = relaxation * self.qp_clbf_relaxation_penalty
        loss["QP relaxation term"] = relaxation_term.mean()

        # Add a loss term for the control input magnitude
        u, _ = self.u(x)
        u_nominal = self.dynamics_model.u_nominal(x)
        controller_squared_difference = ((u - u_nominal) ** 2).sum(dim=-1)
        loss["Control effort difference from nominal"] = (
            self.control_loss_weight * controller_squared_difference.mean()
        )

        return loss

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Get the various losses
        component_losses = {}
        clbf_loss_dict = self.clbf_loss(x, goal_mask, safe_mask, unsafe_mask)
        component_losses.update(clbf_loss_dict)
        controller_loss_dict = self.controller_loss(x)
        component_losses.update(controller_loss_dict)

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        batch_dict = {"loss": total_loss, **component_losses}

        return batch_dict

    def training_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Compute the average loss over all batches for each component
        avg_losses = {}
        for loss_key in outputs[0].keys():
            avg_losses[loss_key] = torch.stack(
                [x[loss_key] for x in outputs if not torch.isnan(x[loss_key])]
            ).mean()

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
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Get the various losses
        component_losses = {}
        clbf_loss_dict = self.clbf_loss(x, goal_mask, safe_mask, unsafe_mask)
        component_losses.update(clbf_loss_dict)
        controller_loss_dict = self.controller_loss(x)
        component_losses.update(controller_loss_dict)

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        batch_dict = {"val_loss": total_loss, **component_losses}

        return batch_dict

    def validation_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Compute the average loss over all batches for each component
        avg_losses = {}
        for loss_key in outputs[0].keys():
            losses = [x[loss_key] for x in outputs if not torch.isnan(x[loss_key])]
            if len(losses) > 0:
                avg_losses[loss_key] = torch.stack(losses).mean()

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
        # at the end of every validation epoch, using arbitrary plotting callbacks!

        for plot_fn in self.plotting_callbacks:
            plot_name, plot = plot_fn(self)
            self.logger.experiment.add_figure(
                plot_name, plot, global_step=self.current_epoch
            )
        self.logger.experiment.close()
        self.logger.experiment.flush()

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-6
        )
