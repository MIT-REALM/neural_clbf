from typing import Tuple, Dict, List, Optional, Callable
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
import pytorch_lightning as pl
from matplotlib.pyplot import figure

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
        clbf_safety_level: float = 10.0,
        clbf_timestep: float = 0.01,
        control_loss_weight: float = 1e-6,
        learning_rate: float = 1e-3,
        plotting_callbacks: Optional[
            List[Callable[["NeuralrCLBFController"], Tuple[str, figure]]]
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
            clbf_safety_level: safety level set value for the CLBF
            clbf_timestep: the timestep to use in simulating forward Vdot
            control_loss_weight: the weight to apply to the control loss
            learning_rate: the learning rate for SGD
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
        self.clbf_safety_level = clbf_safety_level
        self.clbf_timestep = clbf_timestep
        self.control_loss_weight = control_loss_weight
        self.learning_rate = learning_rate

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

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the CLBF value from the output layer of V_net

        args:
            x: bs x sellf.dynamics_model.n_dims the points at which to evaluate the CLBF
        """
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
        u_batched = u_batched.type_as(x)
        # Get nominal control to compare with
        u_nominal = self.dynamics_model.u_nominal(x)
        for i in range(batch_size):
            # Create an optimization problem to find a good input
            opti = casadi.Opti()
            # The decision variables will be the control inputs
            u = opti.variable(self.dynamics_model.n_controls)

            # The objective is simple: minimize the size of the control input
            u_nominal_np = u_nominal[i, :].squeeze().cpu().numpy()
            opti.minimize(casadi.sumsqr(u - u_nominal_np))

            # Add a constraint for CLBF decrease in each scenario
            for j in range(self.n_scenarios):
                # We need to convert these tensors to numpy to make the constraint
                Lf_V_np = Lf_V[i, j, :].squeeze().cpu().numpy().item()
                Lg_V_np = (
                    Lg_V[i, j, :]
                    .squeeze()
                    .cpu()
                    .numpy()
                    .reshape((1, self.dynamics_model.n_controls))
                )
                V_np = V[i].cpu().item()
                opti.subject_to(Lf_V_np + Lg_V_np @ u + self.clbf_lambda * V_np <= 0.0)

            # Set up the solver
            p_opts = {"expand": True, "print_time": 0}
            s_opts = {"max_iter": 1000, "print_level": 0, "sb": "yes"}
            opti.solver("ipopt", p_opts, s_opts)

            # Solve the QP
            solution = opti.solve()

            # Save the solution
            u_batched[i, :] = torch.tensor(solution.value(u)).type_as(x)

        return u_batched

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
        goal_term = F.relu(V0)
        loss["CLBF goal term"] = goal_term.mean()

        #   2.) V <= safe_level in the safe region
        V_safe = self.V(x[safe_mask])
        safe_clbf_term = F.relu(eps + V_safe - self.clbf_safety_level)
        loss["CLBF safe region term"] = safe_clbf_term.mean()

        #   3.) We still want V(safe) to be maximized (as much as possible given the
        # level and descent conditions)
        safe_mask = torch.logical_and(safe_mask, torch.logical_not(goal_mask))
        V_safe = self.V(x[safe_mask])
        safe_max_term = -0.1 * V_safe
        loss["CLBF safe region maximization term"] = safe_max_term.mean()

        #   4.) V >= safe_level in the unsafe region
        V_unsafe = self.V(x[unsafe_mask])
        unsafe_clbf_term = F.relu(eps + self.clbf_safety_level - V_unsafe)
        loss["CLBF unsafe region term"] = unsafe_clbf_term.mean()

        #   5.) A term to encourage satisfaction of CLBF decrease condition
        # We compute the change in V in two ways:
        #       a) simulating x forward in time and checking if V decreases
        #          in each scenario
        #       b) Linearizing V along f.
        # In both cases we use u_NN, but (b) provides a stronger training signal
        # on u_NN.

        # Start with (5a): CLBF decrease in simulation
        clbf_descent_term_sim = torch.tensor(0.0).type_as(x)
        V = self.V(x)
        u_nn = self.u_NN(x)
        for s in self.scenarios:
            xdot = self.dynamics_model.closed_loop_dynamics(x, u_nn, s)
            x_next = x + self.clbf_timestep * xdot
            V_next = self.V(x_next)
            clbf_descent_term_sim += F.relu(
                eps + V_next - (1 - self.clbf_lambda * self.clbf_timestep) * V
            ).mean()
        loss["CLBF descent term (simulated)"] = clbf_descent_term_sim

        # # Then do (5b): CLBF decrease from linearization in each scenario
        # # (this is pretty slow to compute in practice)
        # Lf_V, Lg_V = self.V_lie_derivatives(x)
        # clbf_descent_term_lin = torch.tensor(0.0).type_as(x)
        # for i in range(self.n_scenarios):
        #     Vdot = Lf_V[:, i, :] + torch.sum(
        #         Lg_V[:, i, :] * u_nn, dim=-1
        #     ).unsqueeze(-1)
        #     clbf_descent_term_lin += F.relu(eps + Vdot + self.clbf_lambda * V).mean()

        # loss["CLBF descent term (linearized)"] = clbf_descent_term_lin

        return loss

    def controller_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute a loss to train the proof controller

        args:
            x: the points at which to evaluate the loss
        returns:
            loss: a dictionary containing the losses in each category
        """
        loss = {}

        # Add a loss term for the control input magnitude
        u_nn = self.u_NN(x)
        u_nominal = self.dynamics_model.u_nominal(x)
        controller_squared_difference = ((u_nn - u_nominal) ** 2).sum(dim=-1)
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
