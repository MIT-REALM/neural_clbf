from typing import Tuple, Dict, List, Optional, Callable
from collections import OrderedDict
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import pytorch_lightning as pl
from matplotlib.pyplot import figure

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.utils import Controller
from neural_clbf.experiments.common.episodic_datamodule import EpisodicDataModule


class NeuralSIDCLBFController(pl.LightningModule):
    """
    A neural rCLBF controller that uses a NN to learn the underlying system dynamics as
    well.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        datamodule: EpisodicDataModule,
        clbf_hidden_layers: int = 1,
        clbf_hidden_size: int = 8,
        u_nn_hidden_layers: int = 1,
        u_nn_hidden_size: int = 8,
        f_nn_hidden_layers: int = 1,
        f_nn_hidden_size: int = 8,
        g_nn_hidden_layers: int = 1,
        g_nn_hidden_size: int = 8,
        clbf_lambda: float = 1.0,
        safety_level: float = 2.0,
        discrete_timestep: Optional[float] = None,
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
            u_nn_hidden_layers: number of hidden layers to use for the controller
            u_nn_hidden_size: number of neurons per hidden layer in the controller
                              (control-independent part)
            f_nn_hidden_layers: number of hidden layers to use for the learned dynamics
                                (control-independent part)
            f_nn_hidden_size: number of neurons per hidden layer in the learned dynamics
            g_nn_hidden_layers: number of hidden layers to use for the learned dynamics
                                (control part)
            g_nn_hidden_size: number of neurons per hidden layer in the learned dynamics
                              (control part)
            clbf_lambda: convergence rate for the CLBF
            safety_level: safety level set value for the CLBF
            discrete_timestep: the duration of one discrete time step. Defaults to
                               dynamics_model.dt
            primal_learning_rate: the learning rate for SGD for the network weights
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
        assert clbf_lambda <= 1.0
        self.clbf_lambda = clbf_lambda
        self.safety_level = safety_level
        self.primal_learning_rate = primal_learning_rate
        self.epochs_per_episode = epochs_per_episode

        if discrete_timestep is None:
            self.discrete_timestep = self.dynamics_model.dt
        else:
            self.discrete_timestep = discrete_timestep

        # Some of the dimensions might represent angles. We want to replace these
        # dimensions with two dimensions: sin and cos of the angle. To do this, we need
        # to figure out how many numbers are in the expanded state
        # n_angles = len(self.dynamics_model.angle_dims)
        # self.n_dims_extended = self.dynamics_model.n_dims + n_angles
        # Temporarily disabled
        self.n_dims_extended = self.dynamics_model.n_dims

        # Compute and save the center and range of the state variables
        x_max, x_min = dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_semi_range = (x_max - x_min) / 2.0
        # Angles don't need to be normalized beyond converting to sine/cosine
        self.x_center[self.dynamics_model.angle_dims] = 0.0
        self.x_semi_range[self.dynamics_model.angle_dims] = 1.0

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
            self.n_dims_extended, self.clbf_hidden_size
        )
        self.V_layers["input_layer_activation"] = nn.ReLU()
        for i in range(self.clbf_hidden_layers):
            self.V_layers[f"layer_{i}"] = nn.Linear(
                self.clbf_hidden_size, self.clbf_hidden_size
            )
            self.V_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.V_layers["output_layer"] = nn.Linear(
            self.clbf_hidden_size, self.clbf_hidden_size
        )
        self.V_net = nn.Sequential(self.V_layers)

        # Also define the controller network, denoted u_NN
        # to give the mean and standard deviation of control inputs, respectively
        self.u_nn_hidden_layers = u_nn_hidden_layers
        self.u_nn_hidden_size = u_nn_hidden_size
        # Likewise, build the network up layer by layer, starting with the input
        self.u_NN_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.u_NN_layers["input_layer"] = nn.Linear(
            self.n_dims_extended, self.u_nn_hidden_size
        )
        self.u_NN_layers["input_layer_activation"] = nn.ReLU()
        for i in range(self.u_nn_hidden_layers):
            self.u_NN_layers[f"layer_{i}"] = nn.Linear(
                self.u_nn_hidden_size, self.u_nn_hidden_size
            )
            self.u_NN_layers[f"layer_{i}_activation"] = nn.ReLU()
        # Output layer with ReLU, so the control saturates at [-1, 1]
        self.u_NN_layers["output_layer"] = nn.Linear(
            self.u_nn_hidden_size, self.dynamics_model.n_controls
        )
        # Add an output activation if we're using Tanh
        # self.u_NN_layers["output_layer_activation"] = nn.ReLU()
        self.u_NN = nn.Sequential(self.u_NN_layers)
        # This neural network will represent the mean action, but we also want to
        # keep the log standard deviation as a parameter as well
        u_log_std_init = -0.5 * torch.ones(self.dynamics_model.n_controls)
        self.u_log_std = torch.nn.Parameter(u_log_std_init)

        # Also define the dynamics learning networks, denoted f_nn and g_nn,
        # to learn control-affine dynamics of the form
        #       x_next = x_now + f_nn(x) + g_nn(x) @ u
        self.f_nn_hidden_layers = f_nn_hidden_layers
        self.f_nn_hidden_size = f_nn_hidden_size
        # Likewise, build the network up layer by layer, starting with the input
        self.f_NN_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.f_NN_layers["input_layer"] = nn.Linear(
            self.n_dims_extended,
            self.f_nn_hidden_size,
        )
        self.f_NN_layers["input_layer_activation"] = nn.ReLU()
        for i in range(self.f_nn_hidden_layers):
            self.f_NN_layers[f"layer_{i}"] = nn.Linear(
                self.f_nn_hidden_size, self.f_nn_hidden_size
            )
            self.f_NN_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.f_NN_layers["output_layer"] = nn.Linear(
            self.f_nn_hidden_size, self.dynamics_model.n_dims
        )
        self.f_NN = nn.Sequential(self.f_NN_layers)

        self.g_nn_hidden_layers = g_nn_hidden_layers
        self.g_nn_hidden_size = g_nn_hidden_size
        # Likewise, build the network up layer by layer, starting with the input
        self.g_NN_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.g_NN_layers["input_layer"] = nn.Linear(
            self.n_dims_extended,
            self.g_nn_hidden_size,
        )
        self.g_NN_layers["input_layer_activation"] = nn.ReLU()
        for i in range(self.g_nn_hidden_layers):
            self.g_NN_layers[f"layer_{i}"] = nn.Linear(
                self.g_nn_hidden_size, self.g_nn_hidden_size
            )
            self.g_NN_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.g_NN_layers["output_layer"] = nn.Linear(
            self.g_nn_hidden_size,
            self.dynamics_model.n_dims * self.dynamics_model.n_controls,
        )
        self.g_NN = nn.Sequential(self.g_NN_layers)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input using the stored center point and range, but don't modify
        angles

        args:
            x: bs x self.dynamics_model.n_dims the points to normalize
        """
        # Scale and offset based on the center and range
        x = (x - self.x_center.type_as(x)) / self.x_semi_range.type_as(x)

        return x

    def normalize_w_angles(self, x: torch.Tensor) -> torch.Tensor:
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

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the CLBF value from the output layer of V_net

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        """
        # Apply the offset and range to normalize about zero
        x = self.normalize(x)

        # Compute the CLBF from the network
        V = self.V_net(x)

        return V

    def u(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the learned controller input from the state x

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the
               controller
        Returns:
            u: the bs x self.dynamics_model.n_controls control output
            u_log_prob: the log probability of u (bs)
        """
        # Apply the offset and range to normalize about zero
        x = self.normalize(x)

        # Compute the control effort using the neural network to get the mean of the
        # distribution
        u_mean = self.u_NN(x)
        # Also get the standard deviations
        u_std = torch.exp(self.u_log_std)
        # Construct a Normal distribution with these parameters, and get the action
        # plus the log probability of that action
        dist = Normal(u_mean, u_std)
        u = dist.sample((x.shape[0],))
        u_log_prob = dist.log_prob(u).sum(axis=-1)

        # Scale to reflect plant actuator limits
        upper_lim, lower_lim = self.dynamics_model.control_limits
        u_center = (upper_lim + lower_lim).type_as(x) / 2.0
        u_semi_range = (upper_lim - lower_lim).type_as(x) / 2.0

        u_scaled = u * u_semi_range + u_center

        return u_scaled, u_log_prob

    def learned_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Computes the learned dynamics from the state x with action u

        i.e. computs x_{t+1} given x_t and u_t

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the
               dynamics
            u: bs x self.dynamics_model.n_controls the inputs for each x
        returns:
            x_{t+1} the anticipated next value of the state
        """
        # Apply the offset and range to normalize the state about zero
        x_normalized = self.normalize(x)

        # Get the dynamics
        f = self.f_NN(x_normalized)
        g = self.g_NN(x_normalized)
        # And reshape g to be a matrix
        g = g.reshape((-1, self.dynamics_model.n_dims))

        # Also make sure u is the right shape
        u = u.reshape((-1, self.dynamics_model.n_controls, 1))

        # Compute the dynamics in control-affine form
        x_next = x + f + torch.bmm(g, u).unsqueeze(-1)

        return x_next

    def forward(self, x):
        """Determine the control input for a given state using a NN

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        u, _ = self.u(x)
        return u

    def V_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        dist_to_goal: torch.Tensor,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF

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
        V0 = self.V(x[goal_mask])
        goal_term = F.relu(eps + V0)
        loss.append(("CLBF goal term", goal_term.mean()))

        #   2.) 0 <= V <= safe_level in the safe region
        V_safe = self.V(x[safe_mask])
        safe_clbf_term = F.relu(eps + V_safe - self.safety_level) + F.relu(eps - V_safe)
        loss.append(("CLBF safe region term", safe_clbf_term.mean()))

        #   2b.) for tuning, V >= dist_to_goal in the safe region
        safe_tuning_term = F.relu(eps + dist_to_goal[safe_mask] - V_safe)
        loss.append(("CLBF tuning term", safe_tuning_term.mean()))

        #   3.) V >= safe_level in the unsafe region
        V_unsafe = self.V(x[unsafe_mask])
        unsafe_clbf_term = F.relu(eps + self.safety_level - V_unsafe)
        loss.append(("CLBF unsafe region term", unsafe_clbf_term.mean()))

        #   4.) A term to encourage satisfaction of CLBF decrease condition
        # Get the current CLBF values
        V = self.V(x)
        # Simulate trajectories forwards using the true dynamics...
        next_num_timesteps = round(self.discrete_timestep / self.dynamics_model.dt)
        x_next = self.simulator_fn(x, next_num_timesteps)[:, -1, :]
        # and get the next CLBF value from the V network
        V_next = self.V(x_next)
        V_change = V_next - self.clbf_lambda * V
        clbf_descent_term = F.relu(eps + V_change).mean()
        loss.append(("CLBF descent term", clbf_descent_term))

        return loss

    def dynamics_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate the loss on the dynamics networks

        args:
            x: the points at which to evaluate the loss
        returns:
            loss: a dictionary containing the losses in each category
        """
        loss = {}

        # Use the f network to predict the next state value
        u, _ = self.u(x)
        x_next_est = self.learned_dynamics(x, u)
        # Get the true next state value by simulating forward
        next_num_timesteps = round(self.discrete_timestep / self.dynamics_model.dt)
        x_next_true = self.simulator_fn(x, next_num_timesteps)[:, -1, :]
        # Regress to learn the true next value
        dynamics_difference = ((x_next_est - x_next_true) ** 2).sum(dim=-1)
        loss["Dynamics MSE"] = dynamics_difference.mean()

        return loss

    def u_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate the loss on the controller

        args:
            x: the points at which to evaluate the loss
        returns:
            loss: a dictionary containing the losses in each category
        """
        loss = {}

        # Get the control input
        u, u_log_prob = self.u(x)
        # Get the next state value with this input
        x_next = self.learned_dynamics(x, u)
        # The goal is to minimize V at the next state
        V_next = self.V(x_next)
        loss["Controller loss"] = (V_next * u_log_prob).mean()

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask, dist_to_goal = batch

        # Compute the losses for this optimizer
        component_losses = {}
        self.most_recent_opt_idx = optimizer_idx
        if self.opt_idx_dict[optimizer_idx] == "CLBF":
            component_losses.update(
                self.V_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
            )
        elif self.opt_idx_dict[optimizer_idx] == "dynamics":
            component_losses.update(self.dynamics_loss(x))
        elif self.opt_idx_dict[optimizer_idx] == "u":
            component_losses.update(self.u_loss(x))

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
            self.V_loss(x, goal_mask, safe_mask, unsafe_mask, dist_to_goal)
        )
        component_losses.update(self.dynamics_loss(x))
        component_losses.update(self.u_loss(x))

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

        for plot_fn in self.plotting_callbacks:
            plot_name, plot = plot_fn(self)
            self.logger.experiment.add_figure(
                plot_name, plot, global_step=self.current_epoch
            )
        self.logger.experiment.close()
        self.logger.experiment.flush()

    @pl.core.decorators.auto_move_data
    def simulator_fn(self, x_init: torch.Tensor, num_steps: int):
        return self.dynamics_model.simulate(
            x_init,
            num_steps,
            self.forward,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.discrete_timestep,
        )

    def on_validation_epoch_end(self):
        """This function is called at the end of every validation epoch"""
        # We want to generate new data at the end of every episode
        if self.current_epoch > 0 and self.current_epoch % self.epochs_per_episode == 0:
            # Use the models simulation function with this controller
            self.datamodule.add_data(self.simulator_fn)

    def configure_optimizers(self):
        V_opt = torch.optim.SGD(
            self.V_net.parameters(),
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        u_opt = torch.optim.SGD(
            list(self.u_NN.parameters()) + list(self.u_log_std),
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        dynamics_opt = torch.optim.SGD(
            list(self.f_NN.parameters()) + list(self.g_NN.parameters()),
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "CLBF", 1: "u", 2: "dynamics"}

        return [V_opt, u_opt, dynamics_opt]
