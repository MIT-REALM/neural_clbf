import itertools
from typing import cast, Tuple, List, Optional
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from neural_clbf.systems import ObservableSystem, PlanarLidarSystem
from neural_clbf.controllers.controller import Controller
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite


class NeuralObsBFController(pl.LightningModule, Controller):
    """
    A neural BF controller that relies on observations. Differs from CBF controllers in
    that it does not solve a QP to get the control input and that the BF and policy are
    functions of the observations instead of state. Instead, it trains a policy network
    to satisfy the barrier function decrease conditions.

    More specifically, the BF controller looks for a h and u such that

    h(safe) < 0
    h(unsafe) > 0
    dh/dt|u <= -alpha h

    This proves forward invariance of the 0-sublevel set of h, and since the safe set is
    a subset of this sublevel set, we prove that the unsafe region is not reachable from
    the safe region.

    The networks will have the following architectures:

    h:
        observations + state -> encoder -> fully-connected layers -> h

    u is determined using a lookahead.

    In both of these, we use the same permutation-invariant encoder
    (inspired by Zengyi's approach to macbf)

    encoder:
        observations -> fully-connected layers -> zero invalid elements -> max_pool -> e
    """

    def __init__(
        self,
        dynamics_model: ObservableSystem,
        datamodule: EpisodicDataModule,
        experiment_suite: ExperimentSuite,
        encoder_hidden_layers: int = 2,
        encoder_hidden_size: int = 48,
        h_hidden_layers: int = 2,
        h_hidden_size: int = 48,
        h_alpha: float = 0.9,
        lookahead_grid_n: int = 10,
        lookahead_dual_penalty: float = 1e2,
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-3,
        epochs_per_episode: Optional[int] = None,
        validation_dynamics_model: Optional[ObservableSystem] = None,
        debug_mode: bool = False,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            datamodule: the DataModule to provide data
            experiment_suite: defines the experiments to run during training
            encoder_hidden_layers: number of hidden layers to use for the encoder
            encoder_hidden_size: number of neurons per hidden layer in the encoder
            h_hidden_layers: number of hidden layers to use for the BF network
            h_hidden_size: number of neurons per hidden layer in the BF network
            h_alpha: convergence rate for the BF
            lookahead_grid_n: the number of points to search along each control
                              dimension for the lookahead control.
            lookahead_dual_penalty: the penalty used to dualize the barrier constraint
                                    in the lookahead search.
            controller_period: the timestep to use in simulating forward Vdot
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the BF decrease loss
            epochs_per_episode: optionally gather additional training data every few
                                epochs. If none, no new data will be gathered.f
            validation_dynamics_model: optionally provide a dynamics model to use during
                                       validation
            debug_mode: if True, print and plot some debug information. Defaults false
        """
        super(NeuralObsBFController, self).__init__(
            dynamics_model=dynamics_model,
            experiment_suite=experiment_suite,
            controller_period=controller_period,
        )
        self.save_hyperparameters()

        # Define this again so that Mypy is happy
        self.dynamics_model = dynamics_model
        # And save the validation model
        self.training_dynamics_model = dynamics_model
        self.validation_dynamics_model = validation_dynamics_model

        # Save the datamodule
        self.datamodule = datamodule

        # Save the experiments suits
        self.experiment_suite = experiment_suite

        # Save the other parameters
        self.primal_learning_rate = primal_learning_rate
        assert h_alpha > 0
        assert h_alpha <= 1
        self.h_alpha = h_alpha
        assert lookahead_grid_n > 0
        self.lookahead_grid_n = lookahead_grid_n
        assert lookahead_dual_penalty >= 0.0
        self.lookahead_dual_penalty = lookahead_dual_penalty
        self.epochs_per_episode = epochs_per_episode
        self.debug_mode = debug_mode

        # ----------------------------------------------------------------------------
        # Define the encoder network
        # ----------------------------------------------------------------------------
        self.o_enc_input_size = self.dynamics_model.obs_dim
        self.encoder_hidden_layers = encoder_hidden_layers
        self.encoder_hidden_size = encoder_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.encoder_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.encoder_layers["input_linear"] = nn.Conv1d(
            self.o_enc_input_size,
            self.encoder_hidden_size,
            kernel_size=1,
        )
        self.encoder_layers["input_activation"] = nn.ReLU()
        for i in range(self.encoder_hidden_layers):
            self.encoder_layers[f"layer_{i}_linear"] = nn.Conv1d(
                self.encoder_hidden_size,
                self.encoder_hidden_size,
                kernel_size=1,
            )
            self.encoder_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.encoder_nn = nn.Sequential(self.encoder_layers)

        # ----------------------------------------------------------------------------
        # Define the BF network, which we denote h
        # ----------------------------------------------------------------------------
        self.h_hidden_layers = h_hidden_layers
        self.h_hidden_size = h_hidden_size
        num_h_inputs = self.encoder_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.h_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.h_layers["input_linear"] = nn.Linear(num_h_inputs, self.h_hidden_size)
        self.h_layers["input_activation"] = nn.ReLU()
        for i in range(self.h_hidden_layers):
            self.h_layers[f"layer_{i}_linear"] = nn.Linear(
                self.h_hidden_size, self.h_hidden_size
            )
            self.h_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.h_layers["output_linear"] = nn.Linear(self.h_hidden_size, 1)
        self.h_nn = nn.Sequential(self.h_layers)

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

    def get_observations(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapper around the dynamics model to get the observations"""
        assert isinstance(self.dynamics_model, ObservableSystem)
        return self.dynamics_model.get_observations(x)

    def approximate_lookahead(
        self, x: torch.Tensor, o: torch.Tensor, u: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrapper around the dynamics model to do approximate lookeahead"""
        assert isinstance(self.dynamics_model, ObservableSystem)
        return self.dynamics_model.approximate_lookahead(x, o, u, dt)

    def encoder(self, o: torch.Tensor):
        """Encode the observations o to a fixed-size representation via a permutation-
        invariant encoding

        args:
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
        returns:
            e: bs x self.encoder_hidden_size encoding of the observations
        """
        # We run the observations through the encoder network, which uses
        # convolutional layers with kernel size 1 to implement a fully connected
        # network that doesn't care what the length of the last dimension of the input
        # is (the same transformation will be applied to each point).
        encoded_full = self.encoder_nn(o)

        # Then max-pool over the last dimension.
        encoded_reduced, _ = encoded_full.max(dim=-1)

        return encoded_reduced

    def h(self, x: torch.Tensor, o: torch.Tensor):
        """Return the BF value for the observations o

        args:
            o: bs x self.dynamics_model.n_dims tensor of state
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
        returns:
            h: bs x 1 tensor of BF values
        """
        # Encode the observations
        encoded_obs = self.encoder(o)

        # Then get the barrier function value
        h = self.h_nn(encoded_obs)

        # Add the learned term as a correction to the minimum distance
        min_dist, _ = o.norm(dim=1).min(dim=-1)
        h += 0.3 - min_dist

        return h

    def u_(self, x: torch.Tensor, o: torch.Tensor, h: torch.Tensor):
        """Return the control input for the observations o and state x.

        args:
            x: bs x self.dynamics_model.n_dims tensor of states, used only for
               evaluating the nominal controller
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
            h: bs x 1 tensor of barrier function values
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control values
        """
        batch_size = x.shape[0]
        # The controller is a one-step lookahead controller that attempts to find a
        # control input close to the nominal control which nevertheless satisfies the
        # barrier function decrease condition.
        #
        # We do this by discretizing the action space and searching over the resulting
        # grid. We use approximate lookahead dynamics to propagate the provided
        # observations forward one step without querying the geometry model.
        # We then select the element from the grid that is closest to the nominal
        # control input while still satisfying the barrier function conditions.

        # Get the nominal control
        u_nominal = self.dynamics_model.u_nominal(x)

        # Create the grid of controls over the action space. We can do this once
        # and use the same grid for all batches.
        upper_limit, lower_limit = self.dynamics_model.control_limits
        search_grid_axes = []
        for idx in range(self.dynamics_model.n_controls):
            search_grid_axis = torch.linspace(
                lower_limit[idx].item() * 2,
                upper_limit[idx].item() * 2,
                self.lookahead_grid_n,
            )
            # Add the option to not do anything
            search_grid_axis = torch.cat(
                (search_grid_axis, self.dynamics_model.u_eq[:, idx])
            )
            if idx == 0:
                search_grid_axis = torch.tensor([0.0])
            search_grid_axes.append(search_grid_axis)

        # This will make an N x n_controls tensor,
        # where N = lookahead_grid_n ^ n_controls
        u_options = torch.cartesian_prod(*search_grid_axes).type_as(x)

        # We now want to track which element of the search grid is best for each
        # row of the batched input. Create a tensor of costs for each option in each
        # batch. We'll dualize the barrier function constraint with the set penalty,
        # and we'll start by pre-computing the L2 norm of each option.
        costs = u_options.norm(dim=-1).reshape(1, -1).expand(batch_size, -1)

        # For each control option, run the approximate lookahead to get the next set
        # of observations and compute the cost based on the barrier function constraint
        # and closeness to the nominal control
        for option_idx in range(u_options.shape[0]):
            # Reshape the control option to match the batch size
            u_option = u_options[option_idx, :].reshape(1, -1)  # 1 x n_controls
            u_option = u_option.expand(batch_size, -1)  # batch_size x n_controls
            u_option = u_nominal + u_option

            # Get the next state and observation from lookahead
            x_next, o_next = self.approximate_lookahead(
                x, o, u_option, self.controller_period
            )
            h_next = self.h(x_next, o_next)

            # Get the violation of the barrier decrease constraint
            dhdt = (h_next - h) / self.controller_period
            barrier_function_violation = F.relu(dhdt + self.h_alpha * h).squeeze()

            # Add the violation to the cost
            costs[:, option_idx].add_(
                self.lookahead_dual_penalty * barrier_function_violation
            )

            if self.debug_mode:
                print("=============")
                print(f"x: {x}")
                print(f"u_option: {u_option}")
                print(f"x_next: {x_next}")
                print(f"dhdt = {dhdt}")
                print(f"bf violation = {barrier_function_violation}")

                fig, ax = plt.subplots()
                dynamics_model = cast("PlanarLidarSystem", self.dynamics_model)
                dynamics_model.scene.plot(ax)
                ax.set_aspect("equal")

                ax.plot(x[:, 0], x[:, 1], "ko")

                lidar_pts = o[0, :, :]
                rotation_mat = torch.tensor(
                    [
                        [torch.cos(x[0, 2]), -torch.sin(x[0, 2])],
                        [torch.sin(x[0, 2]), torch.cos(x[0, 2])],
                    ]
                )
                lidar_pts = rotation_mat @ lidar_pts
                lidar_pts[0, :] += x[0, 0]
                lidar_pts[1, :] += x[0, 1]
                ax.plot(lidar_pts[0, :], lidar_pts[1, :], "k-o")

                ax.plot(x_next[:, 0], x_next[:, 1], "ro")

                lidar_pts = o_next[0, :, :]
                rotation_mat = torch.tensor(
                    [
                        [torch.cos(x_next[0, 2]), -torch.sin(x_next[0, 2])],
                        [torch.sin(x_next[0, 2]), torch.cos(x_next[0, 2])],
                    ]
                )
                lidar_pts = rotation_mat @ lidar_pts
                lidar_pts[0, :] += x_next[0, 0]
                lidar_pts[1, :] += x_next[0, 1]
                ax.plot(lidar_pts[0, :], lidar_pts[1, :], "r-o")

                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                plt.show()

        # Now find the option with the lowest cost for each batch
        best_option_idx = torch.argmin(costs, dim=1)

        u = u_nominal + u_options[best_option_idx]

        # Clamp to make sure we don't violate any control limits
        u = torch.clamp(u, lower_limit, upper_limit)

        return u

    def u(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the control input at a given state. Computes the observations and
        barrier function value at this state before computing the control.
        """
        obs = self.get_observations(x)
        h = self.h(x, obs)
        return self.u_(x, obs, h)

    def forward(self, x):
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        return self.u(x)

    def boundary_loss(
        self,
        x: torch.Tensor,
        o: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the BF due to boundary conditions

        args:
            x: the points at which to evaluate the loss,
            o: the observations at x
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        eps = 1e-1
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        h = self.h(x, o)

        #   2.) h < 0 in the safe region
        h_safe = h[safe_mask]
        safe_violation = F.relu(eps + h_safe)
        safe_h_term = 1e2 * safe_violation.mean()
        loss.append(("BF safe region term", safe_h_term))
        if accuracy:
            safe_h_acc = (safe_violation <= eps).sum() / safe_violation.nelement()
            loss.append(("BF safe region accuracy", safe_h_acc))

        #   3.) h > 0 in the unsafe region
        h_unsafe = h[unsafe_mask]
        unsafe_violation = F.relu(eps - h_unsafe)
        unsafe_h_term = 1e2 * unsafe_violation.mean()
        loss.append(("BF unsafe region term", unsafe_h_term))
        if accuracy:
            unsafe_h_acc = (unsafe_violation <= eps).sum() / unsafe_violation.nelement()
            loss.append(("BF unsafe region accuracy", unsafe_h_acc))

        return loss

    def descent_loss(
        self,
        x: torch.Tensor,
        o: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the BF due to the descent condition

        args:
            x: the points at which to evaluate the loss,
            o: the observations at points x
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []
        eps = 1e-2

        # We'll encourage satisfying the BF conditions by...
        #
        #   1) Getting the change in the barrier function value after one control
        #      period has elapsed, and computing the violation of BF conditions
        #      based on that change.

        # Get the barrier function at this current state
        h_t = self.h(x, o)

        # Get the control input
        u_t = self.u_(x, o, h_t)

        # Propagate the dynamics forward via a zero-order hold for one control period
        x_tplus1 = self.dynamics_model.zero_order_hold(x, u_t, self.controller_period)

        # Get the barrier function at this new state
        o_tplus1 = self.get_observations(x_tplus1)
        h_tplus1 = self.h(x, o_tplus1)

        # The discrete-time barrier function is h(t+1) - h(t) \leq -alpha h(t)
        # which reformulates to h(t+1) - (1 - alpha) h(t) \leq 0
        # However, the gradient of loss wrt u becomes very small here (since it scales
        # with the controller period), so approximating the continuous time condition
        # works a bit better:
        #       dh/dt \leq -alpha h ---> (h(t+1) - h(t)) / dt \leq -alpha h(t)
        dhdt = (h_tplus1 - h_t) / self.controller_period
        barrier_function_violation = dhdt + self.h_alpha * h_t
        barrier_function_violation = F.relu(eps + barrier_function_violation)
        barrier_loss = 1e1 * barrier_function_violation.mean()
        barrier_acc = (barrier_function_violation <= eps).sum() / x.shape[0]

        loss.append(("Barrier descent loss", barrier_loss))

        if accuracy:
            loss.append(("Barrier descent accuracy", barrier_acc))

        return loss

    def losses(self):
        """Return a list of loss functions"""
        return [
            self.boundary_loss,
            self.descent_loss,
        ]

    def accuracies(self):
        """Return a list of loss+accuracy functions"""
        return [
            self.boundary_loss,
            self.descent_loss,
        ]

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch
        # and get the observations for x
        o = self.get_observations(x)

        # Compute the losses
        component_losses = {}
        for loss_fn in self.losses():
            component_losses.update(loss_fn(x, o, goal_mask, safe_mask, unsafe_mask))

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
        x, goal_mask, safe_mask, unsafe_mask = batch
        # and get the observations for x
        o = self.get_observations(x)

        # Get the various losses
        component_losses = {}
        for loss_fn in self.losses():
            component_losses.update(loss_fn(x, o, goal_mask, safe_mask, unsafe_mask))

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        # Also compute the accuracy associated with each loss
        for loss_fn in self.accuracies():
            component_losses.update(
                loss_fn(x, o, goal_mask, safe_mask, unsafe_mask, accuracy=True)
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
        # We automatically run experiments during validation

        self.experiment_suite.run_all_and_log_plots(
            self, self.logger, self.current_epoch
        )

        # Now swap in the validation dynamics model and run the experiments again
        if self.validation_dynamics_model is not None:
            self.dynamics_model = self.validation_dynamics_model
            self.experiment_suite.run_all_and_log_plots(
                self, self.logger, self.current_epoch, "validation"
            )
            self.dynamics_model = self.training_dynamics_model

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
        if (
            self.current_epoch > 0  # don't gather new data if we've just started
            and self.epochs_per_episode is not None
            and self.current_epoch % self.epochs_per_episode == 0
        ):
            # Use the model's simulation function with this controller
            self.datamodule.add_data(self.simulator_fn)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.primal_learning_rate,
        )

        self.opt_idx_dict = {0: "all"}

        return [opt]
