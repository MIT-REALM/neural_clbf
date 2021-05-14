from typing import Tuple, List, Optional, Callable
from collections import OrderedDict
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.controllers.utils import Controller
from neural_clbf.experiments.common.episodic_datamodule import EpisodicDataModule


class NeuralBlackBoxLBFController(pl.LightningModule):
    """
    A neural LBF controller that learns a black-box dynamics model along with an LBF-
    based controller
    """

    controller_period: float

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        datamodule: EpisodicDataModule,
        lbf_hidden_layers: int = 2,
        lbf_hidden_size: int = 256,
        u_nn_hidden_layers: int = 2,
        u_nn_hidden_size: int = 256,
        f_nn_hidden_layers: int = 2,
        f_nn_hidden_size: int = 256,
        lbf_lambda: float = 1.0,
        safety_level: float = 1.0,
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-3,
        epochs_per_episode: int = 5,
        num_controller_init_epochs: int = 1,
        plotting_callbacks: Optional[
            List[Callable[[Controller], Tuple[str, figure]]]
        ] = None,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            lbf_hidden_layers: number of hidden layers to use for the LBF network
            lbf_hidden_size: number of neurons per hidden layer in the LBF network
            u_nn_hidden_layers: number of hidden layers to use for the proof controller
            u_nn_hidden_size: number of neurons per hidden layer in the proof controller
            f_nn_hidden_layers: number of hidden layers to use for the dynamics model
            f_nn_hidden_size: number of neurons per hidden layer in the dynamics model
            lbf_lambda: convergence rate for the LBF
            safety_level: safety level set value for the LBF
            controller_period: the timestep to use in simulating forward Vdot
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the LBF decrease loss
            epochs_per_episode: the number of epochs to include in each episode
            num_controller_init_epochs: the number of epochs to train the controller
                                        network to match nominal
            plotting_callbacks: a list of plotting functions that each take a
                                NeuralCLBFController and return a tuple of a string
                                name and figure object to log
        """
        super().__init__()
        self.save_hyperparameters()

        # Save the provided model
        self.dynamics_model = dynamics_model
        self.scenarios = [self.dynamics_model.nominal_params]

        # Save the datamodule
        self.datamodule = datamodule

        # Save the other parameters
        self.lbf_lambda = lbf_lambda
        self.safe_level = safety_level
        self.unsafe_level = safety_level
        self.controller_period = controller_period
        self.primal_learning_rate = primal_learning_rate
        self.epochs_per_episode = epochs_per_episode
        self.num_controller_init_epochs = num_controller_init_epochs

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
        self.lbf_hidden_layers = lbf_hidden_layers
        self.lbf_hidden_size = lbf_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(
            self.n_dims_extended, self.lbf_hidden_size
        )
        self.V_layers["input_activation"] = nn.Tanh()
        for i in range(self.lbf_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.lbf_hidden_size, self.lbf_hidden_size
            )
            self.V_layers[f"layer_{i}_activation"] = nn.Tanh()
        self.V_layers["output_linear"] = nn.Linear(self.lbf_hidden_size, 1)
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
        # No output layer, so the control saturates at [-1, 1]
        self.u_nn_layers["output_linear"] = nn.Linear(
            self.u_nn_hidden_size, self.dynamics_model.n_controls
        )
        self.u_nn_layers["output_activation"] = nn.Tanh()
        self.u_nn = nn.Sequential(self.u_nn_layers)

        # Also define the dynamics learning network, denoted f_nn
        self.f_nn_hidden_layers = f_nn_hidden_layers
        self.f_nn_hidden_size = f_nn_hidden_size
        # Likewise, build the network up layer by layer, starting with the input
        self.f_nn_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.f_nn_layers["input_linear"] = nn.Linear(
            self.n_dims_extended + self.dynamics_model.n_controls, self.f_nn_hidden_size
        )
        self.f_nn_layers["input_activation"] = nn.Tanh()
        for i in range(self.f_nn_hidden_layers):
            self.f_nn_layers[f"layer_{i}_linear"] = nn.Linear(
                self.f_nn_hidden_size, self.f_nn_hidden_size
            )
            self.f_nn_layers[f"layer_{i}_activation"] = nn.Tanh()
        # No output layer, so the control saturates at [-1, 1]
        self.f_nn_layers["output_linear"] = nn.Linear(
            self.f_nn_hidden_size, self.dynamics_model.n_dims
        )
        self.f_nn = nn.Sequential(self.f_nn_layers)

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
            JV[:, sin_idx, sin_idx] = x[:, cos_idx] / self.x_range[dim].type_as(x)
            JV[:, cos_idx, sin_idx] = -x[:, sin_idx] / self.x_range[dim].type_as(x)

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
        x = self.normalize_with_angles(x)

        # Compute the control effort using the neural network
        u = self.u_nn(x)

        # Scale to reflect plant actuator limits
        upper_lim, lower_lim = self.dynamics_model.control_limits
        u_center = (upper_lim + lower_lim).type_as(x) / 2.0
        u_semi_range = (upper_lim - lower_lim).type_as(x) / 2.0

        u_scaled = u * u_semi_range + u_center

        return u_scaled

    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Computes the learned dynamics dx/dt from the state x with action u

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the
               dynamics
            u: bs x self.dynamics_model.n_controls the inputs for each x
        returns:
            xdot: the anticipated state derivative
        """
        # Apply the offset and range to normalize about zero
        # Do this for both the state...
        x = self.normalize_with_angles(x)
        # And the control effort
        upper_lim, lower_lim = self.dynamics_model.control_limits
        u_center = (upper_lim + lower_lim).type_as(x) / 2.0
        u_semi_range = (upper_lim - lower_lim).type_as(x) / 2.0
        u_scaled = (u - u_center) / u_semi_range

        # Concatenate inputs
        inputs = torch.cat((x, u_scaled), dim=-1)

        # Compute the dynamics effort using the neural network
        x_next = self.f_nn(inputs)

        return x_next

    def Vdot(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Lie derivative of the LBF V along the learned dynamics

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            u: bs x self.dynamics_model.n_controls the inputs for each x
        returns:
            V: bs x 1 tensor of the values of V
            Lf_V: bs x 1 tensor of Lie derivatives of V
                  along f
        """
        # Get the Jacobian of V for each entry in the batch
        V, gradV = self.V_with_jacobian(x)

        # Get the dynamics f
        f = self.f(x, u)

        # Multiply these with the Jacobian to get the Lie derivatives
        Lf_V = torch.bmm(gradV, f.unsqueeze(-1)).squeeze(1)

        # return the Lie derivatives
        return V, Lf_V

    def forward(self, x):
        """Determine the control input for a given state using a learned controller

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        u = self.u(x)
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
        Evaluate the loss on the LBF due to boundary conditions

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
        # #   1.) LBF value should be negative on the goal set.
        V = self.V(x)
        V0 = V[goal_mask]
        goal_region_violation = F.relu(eps + V0)
        goal_term = goal_region_violation.mean()

        #   1b.) LBF should be minimized on the goal point
        V_goal_pt = self.V(self.dynamics_model.goal_point.type_as(x)) + 1e-1
        goal_term += (V_goal_pt ** 2).mean()
        loss.append(("LBF goal term", goal_term))

        #   2.) V <= safe_level in the safe region
        V_safe = V[safe_mask]
        safe_V_too_big = F.relu(eps + V_safe - self.safe_level)
        safe_lbf_term = safe_V_too_big.mean()
        #   2b.) V >= 0 in the safe region minus the goal
        safe_minus_goal_mask = torch.logical_and(
            safe_mask, torch.logical_not(goal_mask)
        )
        V_safe_ex_goal = V[safe_minus_goal_mask]
        safe_V_too_small = F.relu(eps - V_safe_ex_goal)
        safe_lbf_term += safe_V_too_small.mean()
        loss.append(("LBF safe region term", safe_lbf_term))

        #   3.) V >= unsafe_level in the unsafe region
        V_unsafe = V[unsafe_mask]
        unsafe_V_too_small = F.relu(eps + self.unsafe_level - V_unsafe)
        unsafe_lbf_term = unsafe_V_too_small.mean()
        loss.append(("LBF unsafe region term", unsafe_lbf_term))

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
        Evaluate the loss on the LBF due to the descent condition

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
        eps = 0.01

        #   1.) A term to encourage satisfaction of the LBF decrease condition,
        # which requires that V is decreasing everywhere where V <= safe_level

        # Get the control input, the current LBF value, and the derivative of the LBF
        u_nn = self.u(x)
        V, Vdot = self.Vdot(x, u_nn)

        # Figure out where this decrease condition needs to hold
        V = self.V(x)
        condition_active = F.relu(self.safe_level - V)

        # And compute the violation in that region
        violation = F.relu(eps + Vdot + self.lbf_lambda * V) * condition_active
        clbf_descent_term_lin = violation.mean()
        loss.append(("CLBF descent term (linearized)", clbf_descent_term_lin))

        return loss

    def controller_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        dist_to_goal: torch.Tensor,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the controller due to the nominal matching term

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            dist_to_goal: the distance from x to the goal region
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss based on the following...
        loss = []

        #   1.) Compare the controller to the nominal, with a loss that decreases at
        # each epoch
        u_nn = self.u(x)
        u_nominal = self.dynamics_model.u_nominal(x)
        control_mse_loss = (u_nn - u_nominal) ** 2
        control_mse_loss = control_mse_loss.mean()
        epoch_cutoff = max(self.current_epoch - self.num_controller_init_epochs, 0)
        control_mse_loss /= 100 * epoch_cutoff + 1
        loss.append(("Controller MSE", control_mse_loss))

        return loss

    def dynamics_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        dist_to_goal: torch.Tensor,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the learned dynamics based on how well they match
        the black box dynamics.

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            dist_to_goal: the distance from x to the goal region
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss based on the following...
        loss = []

        #   1.) Compare the black box and learned dynamics with on-policy actions
        u_nn = self.u(x)
        learned_dynamics = self.f(x, u_nn)
        true_dynamics = self.dynamics_model.closed_loop_dynamics(x, u_nn)
        dynamics_mse_loss = (learned_dynamics - true_dynamics) ** 2
        dynamics_mse_loss = dynamics_mse_loss.mean()
        loss.append(("Dynamics MSE", dynamics_mse_loss))

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask, dist_to_goal = batch

        # Compute the losses
        component_losses = {}
        if self.opt_idx_dict[optimizer_idx] == "V_u":
            component_losses.update(
                self.descent_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
            )
            component_losses.update(
                self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
            )
            component_losses.update(
                self.controller_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
            )
        elif self.opt_idx_dict[optimizer_idx] == "dynamics":
            component_losses.update(
                self.dynamics_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
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
        # from all of them
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
    def simulator_fn(
        self,
        x_init: torch.Tensor,
        num_steps: int,
    ):
        return self.dynamics_model.simulate(
            x_init,
            num_steps,
            self.u,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.controller_period,
        )

    def on_validation_epoch_end(self):
        """This function is called at the end of every validation epoch"""
        # We want to generate new data at the end of every episode
        if self.current_epoch > 0 and self.current_epoch % self.epochs_per_episode == 0:
            # Use the models simulation function with this controller
            def simulator_fn_wrapper(x_init: torch.Tensor, num_steps: int):
                return self.simulator_fn(
                    x_init,
                    num_steps,
                )

            self.datamodule.add_data(simulator_fn_wrapper)

    def configure_optimizers(self):
        V_u_opt = torch.optim.SGD(
            list(self.V_nn.parameters()) + list(self.u_nn.parameters()),
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        f_opt = torch.optim.SGD(
            self.f_nn.parameters(),
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "V_u", 1: "dynamics"}

        return [V_u_opt, f_opt]
