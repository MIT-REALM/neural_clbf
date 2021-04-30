from typing import Tuple, List, Optional, Callable
from collections import OrderedDict

from qpth.qp import QPFunction
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
        clbf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
        lookahead: float = 0.1,
        primal_learning_rate: float = 1e-3,
        epochs_per_episode: int = 5,
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
            clbf_relaxation_penalty: the penalty for relaxing CLBF conditions.
            controller_period: the timestep to use in simulating forward Vdot
            lookahead: how far to simulate forward to gauge decrease in V
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the CLBF decrease loss
            epochs_per_episode: the number of epochs to include in each episode
            plotting_callbacks: a list of plotting functions that each take a
                                NeuralCLBFController and return a tuple of a string
                                name and figure object to log
        """
        super().__init__()

        # Save the provided model
        self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the datamodule
        self.datamodule = datamodule

        # Save the other parameters
        self.clbf_lambda = clbf_lambda
        self.safe_level = safety_level
        self.unsafe_level = safety_level
        self.clbf_relaxation_penalty = clbf_relaxation_penalty
        self.controller_period = controller_period
        self.lookahead = lookahead
        self.primal_learning_rate = primal_learning_rate
        self.epochs_per_episode = epochs_per_episode

        # Compute and save the center and range of the state variables
        x_max, x_min = dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_range = (x_max - x_min) / 2.0
        # Scale to get the input between (-k, k), centered at 0
        k = 10.0
        self.x_range = self.x_range / k

        # Get plotting callbacks
        if plotting_callbacks is None:
            plotting_callbacks = []
        self.plotting_callbacks = plotting_callbacks

        # Define the CLBF network, which we denote V
        self.clbf_hidden_layers = clbf_hidden_layers
        self.clbf_hidden_size = clbf_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(
            self.dynamics_model.n_dims, self.clbf_hidden_size
        )
        self.V_layers["input_activation"] = nn.Tanh()
        for i in range(self.clbf_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.clbf_hidden_size, self.clbf_hidden_size
            )
            self.V_layers[f"layer_{i}_activation"] = nn.Tanh()
        self.V_layers["output_linear"] = nn.Linear(self.clbf_hidden_size, 1)
        self.V_nn = nn.Sequential(self.V_layers)

        # Also define the proof controller network, denoted u_nn
        self.u_nn_hidden_layers = u_nn_hidden_layers
        self.u_nn_hidden_size = u_nn_hidden_size
        # Likewise, build the network up layer by layer, starting with the input
        self.u_NN_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.u_NN_layers["input_linear"] = nn.Linear(
            self.dynamics_model.n_dims, self.u_nn_hidden_size
        )
        self.u_NN_layers["input_activation"] = nn.Tanh()
        for i in range(self.u_nn_hidden_layers):
            self.u_NN_layers[f"layer_{i}_linear"] = nn.Linear(
                self.u_nn_hidden_size, self.u_nn_hidden_size
            )
            self.u_NN_layers[f"layer_{i}_activation"] = nn.Tanh()
        # No output layer, so the control saturates at [-1, 1]
        self.u_NN_layers["output_linear"] = nn.Linear(
            self.u_nn_hidden_size, self.dynamics_model.n_controls
        )
        self.u_NN_layers["output_activation"] = nn.Tanh()
        self.u_NN = nn.Sequential(self.u_NN_layers)

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

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # Apply the offset and range to normalize about zero
        x = self.normalize(x)

        # Compute the CLBF layer-by-layer, computing the Jacobian alongside

        # We need to initialize the Jacobian to reflect the normalization that's already
        # been done to x
        bs = x.shape[0]
        JV = torch.zeros((bs, x.shape[-1], x.shape[-1])).type_as(x)
        # and for each dimension, we need to scale by the normalization
        for dim in range(self.dynamics_model.n_dims):
            JV[:, dim, dim] = 1.0 / self.x_range[dim].type_as(x)

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
        x = self.normalize(x)

        # Compute the control effort using the neural network
        u = self.u_NN(x)

        # Scale to reflect plant actuator limits
        upper_lim, lower_lim = self.dynamics_model.control_limits
        u_center = (upper_lim + lower_lim).type_as(x) / 2.0
        u_semi_range = (upper_lim - lower_lim).type_as(x) / 2.0

        u_scaled = u * u_semi_range + u_center

        return u_scaled

    def V_lie_derivatives(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Lie derivatives of the CLBF V along the control-affine dynamics

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            Lf_V: bs x self.n_scenarios x 1 tensor of Lie derivatives of V
                  along f
            Lg_V: bs x self.n_scenarios x self.dynamics_model.n_controls tensor
                  of Lie derivatives of V along g
        """
        # Get the Jacobian of V for each entry in the batch
        _, gradV = self.V_with_jacobian(x)

        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, self.n_scenarios, 1)
        Lg_V = torch.zeros(batch_size, self.n_scenarios, self.dynamics_model.n_controls)
        Lf_V = Lf_V.type_as(x)
        Lg_V = Lg_V.type_as(x)

        for i in range(self.n_scenarios):
            # Get the dynamics f and g for this scenario
            s = self.scenarios[i]
            f, g = self.dynamics_model.control_affine_dynamics(x, params=s)

            # Multiply these with the Jacobian to get the Lie derivatives
            Lf_V[:, i, :] = torch.bmm(gradV, f).squeeze(1)
            Lg_V[:, i, :] = torch.bmm(gradV, g).squeeze(1)

        # return the Lie derivatives
        return Lf_V, Lg_V

    def solve_CLBF_QP(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLBF had to be relaxed in each
                        case
            objectives: bs x 1 tensor of the QP objective.
        """
        # Get the value of the CLBF and its Lie derivatives
        V = self.V(x)
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
        #           ||u - u_nominal||^2 + relaxation_penalty * r^2
        #
        # This reduces to (ignoring constant terms)
        #
        #           u^T I u - 2 u_nominal^T u + relaxation_penalty * r^2
        #
        # so we need (since qpth optimizes 1/2 x^T Q x + p^T x)
        #
        #           Q = 2 * [I 0
        #                    0 relaxation_penalty]
        #           p = [-2 u_nominal^T 0]
        #
        # Expressing the constraints formally:
        #
        #       Gz <= h
        #
        # where h = [-L_f V - lambda V, 0, h_u]^T and G = [L_g V, -1
        #                                                  ...repeated for each scenario
        #                                                  0,     -1
        #                                                  G_u,    0]
        # We can optionally add the user-specified inequality constraints as G_u
        n_controls = self.dynamics_model.n_controls
        n_scenarios = self.n_scenarios
        bs = x.shape[0]

        # Start by building the cost
        Q = torch.zeros(bs, n_controls + 1, n_controls + 1).type_as(x)
        for j in range(n_controls):
            Q[:, j, j] = 1.0
        Q[:, -1, -1] = self.clbf_relaxation_penalty + 0.01
        Q *= 2.0
        p = torch.zeros(bs, n_controls + 1).type_as(x)
        u_nominal = self.dynamics_model.u_nominal(x)
        p[:, :-1] = -2.0 * u_nominal

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
        r = result[:, n_controls:]

        # Get the objective
        Qx = torch.bmm(Q, result.unsqueeze(-1))
        xQx = torch.bmm(result.unsqueeze(1), Qx)
        px = torch.bmm(p.unsqueeze(1), result.unsqueeze(-1))
        objective = 0.5 * xQx + px
        # Also add the constant nominal control term to put the unconstrained minimum
        # objective at zero
        objective = objective.reshape(-1, 1) + (u_nominal ** 2).sum(dim=-1)

        return u.type_as(x), r.type_as(x), objective.type_as(x)

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
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to boundary conditions

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            dist_to_goal: the distance from x to the goal region
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        eps = 1e-2
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []
        #   1.) CLBF value should be negative on the goal set.
        V = self.V(x)
        V0 = V[goal_mask]
        goal_region_violation = F.relu(eps + V0)
        goal_term = goal_region_violation.mean()

        #   1b.) CLBF should be minimized on the goal point
        V_goal_pt = self.V(self.dynamics_model.goal_point) + 1e-1
        goal_term += (V_goal_pt ** 2).mean()
        loss.append(("CLBF goal term", goal_term))

        #   2.) V <= safe_level in the safe region
        V_safe = V[safe_mask]
        safe_V_too_big = F.relu(eps + V_safe - self.safe_level)
        safe_clbf_term = safe_V_too_big.mean()
        #   2b.) V >= 0 in the safe region minus the goal
        safe_minus_goal_mask = torch.logical_and(
            safe_mask, torch.logical_not(goal_mask)
        )
        V_safe_ex_goal = V[safe_minus_goal_mask]
        safe_V_too_small = F.relu(eps - V_safe_ex_goal)
        safe_clbf_term += safe_V_too_small.mean()
        loss.append(("CLBF safe region term", safe_clbf_term))

        # #   2c.) for tuning, V >= dist_to_goal in the safe region
        # safe_tuning_term = F.relu(eps + dist_to_goal[safe_mask] - V_safe)
        # loss.append(("CLBF tuning term", safe_tuning_term.mean()))

        #   3.) V >= unsafe_level in the unsafe region
        V_unsafe = V[unsafe_mask]
        unsafe_V_too_small = F.relu(eps + self.unsafe_level - V_unsafe)
        unsafe_clbf_term = unsafe_V_too_small.mean()
        loss.append(("CLBF unsafe region term", 10 * unsafe_clbf_term))

        return loss

    def descent_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        dist_to_goal: torch.Tensor,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to the descent condition

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            dist_to_goal: the distance from x to the goal region
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        #   1.) A term to encourage satisfaction of the CLBF decrease condition,
        # by minimizing the relaxation in the CLBF conditions needed to solve the QP
        u, relax, objective = self.solve_CLBF_QP(x)
        # relax_term = self.clbf_relaxation_penalty * relax.mean()
        relax_term = relax.mean()
        loss.append(("CLBF QP relaxation", relax_term))

        # #   2.) Also minimize the objective of the QP. This should push in mostly the
        # # same direction as (1), since the objective includes a relax^2 term.
        # objective_term = objective.mean()
        # loss.append(("CLBF QP objective", objective_term))

        #   3.) A term to encourage satisfaction of CLBF decrease condition
        # We compute the change in V by simulating x forward in time and checking if V
        # decreases
        V = self.V(x)
        clbf_descent_term_sim = torch.tensor(0.0).type_as(x)
        for s in self.scenarios:
            sim_timesteps = round(self.lookahead / self.dynamics_model.dt)
            x_next = self.simulator_fn(x, sim_timesteps, use_qp=True)[:, -1, :]
            V_next = self.V(x_next)
            # dV/dt \approx (V_next - V)/dt + lambda V \leq 0
            # simplifies to V_next - V + dt * lambda V \leq 0
            # simplifies to V_next - (1 - dt * lambda) V \leq 0
            clbf_descent_term_sim += F.relu(
                V_next - (1 - self.clbf_lambda * self.lookahead) * V
            ).mean()
        loss.append(("CLBF descent term (simulated)", clbf_descent_term_sim))

        # #   4b.) A term to encourage satisfaction of CLBF decrease condition
        # # This time, we compute the decrease using linearization, which provides a
        # # training signal for the controller
        # clbf_descent_term_lin = torch.tensor(0.0).type_as(x)
        # # Get the current value of the CLBF and its Lie derivatives
        # # (Lie derivatives are computed using a linear fit of the dynamics)
        # # TODO @dawsonc do we need dynamics learning here?
        # Lf_V, Lg_V = self.V_lie_derivatives(x)
        # # Get the control and reshape it to bs x n_controls x 1
        # u_nn = self.u(x)
        # u_nn = u_nn.unsqueeze(-1)
        # for i, s in enumerate(self.scenarios):
        #     # Use these dynamics to compute the derivative of V
        #     Vdot = Lf_V[:, i, :] + torch.bmm(Lg_V[:, i, :].unsqueeze(1), u_nn)
        #     clbf_descent_term_lin += F.relu(eps + Vdot + self.clbf_lambda * V).mean()
        # loss.append(("CLBF descent term (linearized)", clbf_descent_term_lin))

        return loss

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask, dist_to_goal = batch

        # Compute the losses
        component_losses = {}
        component_losses.update(
            self.descent_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
        )
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
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
        # at the end of every validation epoch, using arbitrary plotting callbacks!

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

    @pl.core.decorators.auto_move_data
    def simulator_fn(self, x_init: torch.Tensor, num_steps: int, use_qp: bool = True):
        if use_qp:
            controller_fn = self.forward
        else:
            controller_fn = self.u

        return self.dynamics_model.simulate(
            x_init,
            num_steps,
            controller_fn,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.controller_period,
        )

    def on_validation_epoch_end(self):
        """This function is called at the end of every validation epoch"""
        # We want to generate new data at the end of every episode
        if self.current_epoch > 0 and self.current_epoch % self.epochs_per_episode == 0:
            # Use the models simulation function with this controller
            def simulator_fn(x_init: torch.Tensor, num_steps: int):
                return self.dynamics_model.simulate(x_init, num_steps, self.u)

            self.datamodule.add_data(simulator_fn)

    def configure_optimizers(self):
        primal_opt = torch.optim.SGD(
            list(self.V_nn.parameters()) + list(self.u_NN.parameters()),
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        # self.opt_idx_dict = {0: "descent", 1: "boundary"}

        return [primal_opt]
