import itertools
from typing import cast, Tuple, List, Optional
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from neural_clbf.systems import ObservableSystem, PlanarLidarSystem  # noqa
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

    u is determined using a lookahead with a hand-designed lyapunov function (LF)

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
        V_hidden_layers: int = 2,
        V_hidden_size: int = 48,
        V_lambda: float = 0.0,
        lookahead_grid_n: int = 10,
        lookahead_dual_penalty: float = 1e2,
        V_goal_tolerance: float = 0.7,
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-3,
        epochs_per_episode: Optional[int] = None,
        validation_dynamics_model: Optional[ObservableSystem] = None,
        debug_mode: bool = False,
        state_only: bool = False,
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
            h_hidden_layers: number of hidden layers to use for the LF network
            h_hidden_size: number of neurons per hidden layer in the LF network
            V_lambda: convergence rate for the LF
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
            state_only: if True, define the barrier function in terms of robot state
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
        assert V_lambda >= 0
        assert V_lambda <= 1
        self.V_lambda = V_lambda
        assert lookahead_grid_n > 0
        self.lookahead_grid_n = lookahead_grid_n
        assert lookahead_dual_penalty >= 0.0
        self.lookahead_dual_penalty = lookahead_dual_penalty
        self.epochs_per_episode = epochs_per_episode
        self.debug_mode_exploratory = debug_mode
        self.debug_mode_goal_seeking = debug_mode
        self.state_only = state_only

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
        if self.state_only:
            # If we're in "state-only" mode, take state as the input to h
            # (this allows training to compare with a state-based CBF)
            num_h_inputs = self.dynamics_model.n_dims
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

        # ----------------------------------------------------------------------------
        # Define the LF network, which we denote V
        # ----------------------------------------------------------------------------
        self.V_hidden_layers = V_hidden_layers
        self.V_hidden_size = V_hidden_size
        # For turtlebot, the inputs to V are range and heading to the origin
        num_V_inputs = 3
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(num_V_inputs, self.V_hidden_size)
        self.V_layers["input_activation"] = nn.ReLU()
        for i in range(self.V_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.V_hidden_size, self.V_hidden_size
            )
            if i < self.V_hidden_layers - 1:
                self.V_layers[f"layer_{i}_activation"] = nn.ReLU()
        # Use the positive definite trick to encode V
        # self.V_layers["output_linear"] = nn.Linear(self.V_hidden_size, 1)
        self.V_nn = nn.Sequential(self.V_layers)

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
            x: bs x self.dynamics_model.n_dims tensor of state
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
        returns:
            h: bs x 1 tensor of BF values
        """
        # Then get the barrier function value.
        # Most of the time, this is done based on observations, but we want to
        # enable comparisons with a state-based barrier function, so we include the
        # option for that as well.
        if not self.state_only:
            # Encode the observations
            encoded_obs = self.encoder(o)

            # Get the barrier function based on those observations
            h = self.h_nn(encoded_obs)

            # Add the learned term as a correction to the minimum distance
            min_dist, _ = o.norm(dim=1).min(dim=-1)
            min_dist = min_dist.reshape(-1, 1)
            h += 0.3 - min_dist
        else:
            h = self.h_nn(x)

        return h

    def V(self, x: torch.Tensor):
        """Return the LF value for state x

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            V: bs x 1 tensor of BF values
        """
        # The inputs to V for the turtlebot are range and heading to the origin
        range_to_goal = torch.sqrt((x[:, :2] ** 2).sum(dim=-1)).reshape(-1, 1)
        # Phi is the angle from the current heading towards the origin
        angle_from_bot_to_origin = torch.atan2(-x[:, 1], -x[:, 0])
        theta = x[:, 2]
        phi = angle_from_bot_to_origin - theta
        # First, wrap the angle error into [-pi, pi]
        phi = torch.atan2(torch.sin(phi), torch.cos(phi))
        phi = phi.reshape(-1, 1)

        V_input = torch.hstack((range_to_goal, torch.cos(phi), torch.sin(phi)))
        V = self.V_nn(V_input)
        V = 0.5 * (V ** 2).sum(dim=-1).reshape(-1, 1)

        # Add the learned term as a correction to a norm-like base
        distance_squared = range_to_goal ** 2
        V += 1.0 * distance_squared + 0.5 * (1 - torch.cos(phi))

        return V

    GOAL_SEEKING_MODE = 0
    EXPLORATORY_MODE = 1

    def reset_controller(self, x):
        """Initialize the state-machine controller at states x.

        args:
            x: bs x self.dynamics_model.n_dims tensor of states
        """
        batch_size = x.shape[0]

        # This tensor indicates which mode each row of x is in
        self.controller_mode = torch.zeros(batch_size, dtype=torch.int).type_as(x)
        self.controller_mode += self.GOAL_SEEKING_MODE
        self.last_controller_mode = self.controller_mode.clone().detach()

        # Store the LF and BF at each hit point
        self.hit_points_V = torch.zeros(batch_size, 1).type_as(x).detach()
        self.hit_points_h = torch.zeros(batch_size, 1).type_as(x).detach()

        # This tensor stores number of steps spent in the exploratory mode
        self.num_exploratory_steps = torch.zeros(batch_size, 1).type_as(x).detach()
        self.num_exploratory_steps.requires_grad = False

        # Also store the previous control input to avoid being too jerky during the
        # exploratory phase
        self.u_prev = torch.zeros(batch_size, self.dynamics_model.n_controls).type_as(x)

    def switch_modes(self, destination_mode, switch_flags):
        """Switch modes as needed

        args:
            destination_mode: an int representing the mode to switch to
            switch_flags: a self.controller_mode.shape[0] boolean tensor that is true
                          for every row that should switch modes to destination_mode
        """
        self.controller_mode[switch_flags] = destination_mode

    def u_(
        self, x: torch.Tensor, o: torch.Tensor, h: torch.Tensor, V: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the control input for the observations o and state x

        args:
            x: bs x self.dynamics_model.n_dims tensor of states
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
            h: bs x 1 tensor of barrier function values
            V: bs x 1 tensor of lyapunov function values
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control values
            cost: bs tensor of cost for each control action
        """
        # Make sure the input size hasn't changed
        batch_size = x.shape[0]
        assert batch_size == self.controller_mode.shape[0], "Batch size changed!"

        # Get the controller mode for each batch
        goal_seeking = self.controller_mode == self.GOAL_SEEKING_MODE
        exploratory = self.controller_mode == self.EXPLORATORY_MODE

        # Update the number of steps in the exploratory mode
        self.num_exploratory_steps[exploratory] += 1.0

        # Both control modes do a lookeahead that wants a list of options for control
        # inputs, and the next state, observations, V, and h values for each option.
        # We can save time by doing this once for both modes
        u_options, x_next, o_next, h_next, V_next, idxs = self.lookahead(x, o)

        # Expand h and V to match the size of h_next and V_next
        h = h[idxs[:, 0]]
        V = V[idxs[:, 0]]

        # Get control inputs for the goal seeking and exploratory modes,
        # and at the same time see if any need to switch modes
        u_goal_seeking, gs_cost, switch_to_exploratory = self.u_goal_seeking(
            x, o, h, V, idxs, u_options, x_next, o_next, h_next, V_next
        )
        u_exploratory, exp_cost, switch_to_goal_seeking = self.u_exploratory(
            x, o, h, V, idxs, u_options, x_next, o_next, h_next, V_next
        )

        # Collate the control inputs and costs
        u = torch.zeros_like(u_goal_seeking).type_as(x)
        costs = torch.zeros_like(gs_cost).type_as(x)
        u[goal_seeking] = u_goal_seeking[goal_seeking]
        u[exploratory] = u_exploratory[exploratory]
        u = u.reshape(x.shape[0], -1)
        costs[goal_seeking] = gs_cost[goal_seeking]
        costs[exploratory] = exp_cost[exploratory]

        # Switch modes as needed
        self.switch_modes(self.GOAL_SEEKING_MODE, switch_to_goal_seeking)
        self.switch_modes(self.EXPLORATORY_MODE, switch_to_exploratory)

        # if (self.last_controller_mode != self.controller_mode).any():
        #     print(self.controller_mode)
        self.last_controller_mode = self.controller_mode.clone().detach()

        # Save the control as the previous control
        self.u_prev = u

        # Return the correct control
        return u, costs

    def lookahead(
        self,
        x: torch.Tensor,
        o: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Generate a list of control input options and evaluate them.

        args:
            x: bs x self.dynamics_model.n_dims tensor of states
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
        returns:
            u_options: N x n_controls tensor of possible control inputs, where
                       N = self.lookahead_grid_n ^ n_controls (same below)
            x_next: bs * N x self.dynamics_model.n_dims tensor of next states
            o_next: bs * N x self.dynamics_model.obs_dim x self.dynamics_model.n_obs
                    tensor of observations at the next state
            h_next: bs * N x 1 tensor of barrier function values at the next state
            V_next: bs * N x 1 tensor of Lyapunov function values at the next state
            idxs: bs * N x 2 tensor of indices into x and u_options
        """
        # Create the grid of controls over the action space.
        upper_lim, lower_lim = self.dynamics_model.control_limits
        search_grid_axes = []
        for idx in range(self.dynamics_model.n_controls):
            search_grid_axis = torch.linspace(
                lower_lim[idx].item(),
                upper_lim[idx].item(),
                self.lookahead_grid_n,
            )
            # Add the option to not do anything
            search_grid_axis = torch.cat((search_grid_axis, torch.tensor([0.0])))
            search_grid_axes.append(search_grid_axis)

        # This will make an N x n_controls tensor,
        # where N = lookahead_grid_n ^ n_controls
        u_options = torch.cartesian_prod(*search_grid_axes).type_as(x)
        u_options = torch.unique(u_options, dim=0)

        # We want to move forward, so modify the controls
        u_options[:, 0] += 0.1  # bias v forward for turtlebot

        if self.debug_mode_goal_seeking or self.debug_mode_exploratory:
            print("u axes")
            for u_ax in search_grid_axes:
                print(f"\t{u_ax.T}")
            print("u options")
            print(u_options.T)

        # Combine the control options with the states so we can run the lookahead in
        # a batch
        x_indices = torch.arange(x.shape[0])
        u_indices = torch.arange(u_options.shape[0])
        idxs = torch.cartesian_prod(x_indices, u_indices)

        # Run the approximate lookahead to get the next set of states and observations,
        # from which we'll get the LF, and BF
        x_nexts, o_nexts = self.approximate_lookahead(
            x[idxs[:, 0]],
            o[idxs[:, 0]],
            u_options[idxs[:, 1]],
            self.controller_period,
        )
        V_nexts = self.V(x_nexts)
        h_nexts = self.h(x_nexts, o_nexts)

        return u_options, x_nexts, o_nexts, h_nexts, V_nexts, idxs

    def u_goal_seeking(
        self,
        x: torch.Tensor,
        o: torch.Tensor,
        h: torch.Tensor,
        V: torch.Tensor,
        idxs: torch.Tensor,
        u_options: torch.Tensor,
        x_nexts: torch.Tensor,
        o_nexts: torch.Tensor,
        h_nexts: torch.Tensor,
        V_nexts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the control input for the observations o and state x in the goal
        seeking mode.

        args:
            x: bs x self.dynamics_model.n_dims tensor of states
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
            h: bs x 1 tensor of barrier function values
            V: bs x 1 tensor of Lyapunov function values
            idxs: bs * N x 2 tensor of indices into *_nexts and u_options
            u_options: N x n_controls tensor of possible control inputs, where
                       N = self.lookahead_grid_n ^ n_controls (same below)
            x_nexts: bs * N x self.dynamics_model.n_dims tensor of next states
            o_nexts: bs * N x self.dynamics_model.obs_dim x self.dynamics_model.n_obs
                     tensor of observations at the next state
            h_nexts: bs * N x 1 tensor of barrier function values at the next state
            V_nexts: bs * N x 1 tensor of Lyapunov function values at the next state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control values
            cost: bs tensor of cost for each control action
            switch_to_exploratory: bs tensor of booleans indicating which rows should
                                   switch controller modes to exploratory
        """
        # The controller is a one-step lookahead controller that attempts to find a
        # control input that decreases the Lyapunov function and satisfies the
        # barrier function decrease condition.
        #
        # We do this by discretizing the action space and searching over the resulting
        # grid. We use approximate lookahead dynamics to propagate the provided
        # observations forward one step without querying the geometry model.
        #
        # We then select the control input that minimizes
        #
        #       Q * (V_{t+1} - (1 - lambda) V_t) + R ||u||^2
        # s.t.
        #
        #       h_{t+1} - (1 - alpha) h_t <= 0
        #
        # where the constraint is relaxed to a cost with a large penalty
        Q = 1.0
        R = 0.01

        # We now want to track which element of the search grid is best for each
        # row of the batched input. Create a tensor of costs for each option in each
        # batch. We'll dualize the barrier function constraint with the set penalty,
        # and we'll start by pre-computing the L2 norm of each option, scaled by R
        costs = R * u_options[idxs[:, 1]].norm(dim=-1)

        # Add in the barrier function constraint penalty
        costs.add_(
            self.lookahead_dual_penalty
            * F.leaky_relu(
                h_nexts - (1 - self.h_alpha) * h, negative_slope=0.001
            ).squeeze()
        )

        # Also add in the cost term for the LF decrease
        costs.add_(Q * (V_nexts - (1 - self.V_lambda) * V).squeeze())

        # If we're debugging, loop through the options and visualize them
        if self.debug_mode_goal_seeking and torch.allclose(
            self.controller_mode, torch.zeros_like(self.controller_mode)
        ):
            for idx, idx_tensor in enumerate(idxs):
                state_idx, option_idx = idx_tensor
                # Reshape the control option to match the batch size
                u_option = u_options[option_idx, :].reshape(1, -1)  # 1 x n_controls

                # Get the next state and observation from lookahead
                x_next = x_nexts[idx, :].unsqueeze(0)
                o_next = o_nexts[idx, :, :].unsqueeze(0)
                V_next = V_nexts[idx, :].unsqueeze(0)
                h_next = h_nexts[idx, :].unsqueeze(0)

                # Get the violation of the barrier decrease constraint
                barrier_function_violation = F.leaky_relu(
                    h_next - (1 - self.h_alpha) * h, negative_slope=0.001
                ).squeeze()
                # and get the LF decrease
                lyapunov_function_change = V_next - (1 - self.V_lambda) * V
                lyapunov_function_change = lyapunov_function_change.squeeze()

                print("=============")
                print(f"x: {x}")
                print(f"h: {h}")
                print(f"V: {V}")
                print(f"u_option: {u_option}")
                print(f"x_next: {x_next}")
                print(f"h_next: {h_next}")
                print(f"V_next: {V_next}")
                print(f"bf violation = {barrier_function_violation}")
                print(f"lf change = {lyapunov_function_change}")
                print(f"cost = {costs[idx]}")

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
        batch_size = x.shape[0]
        num_options = u_options.shape[0]
        best_option_cost, best_option_idx = torch.min(
            costs.view(batch_size, num_options, -1), dim=1
        )
        u = u_options[best_option_idx]

        # Clamp to make sure we don't violate any control limits
        upper_limit, lower_limit = self.dynamics_model.control_limits
        u = torch.clamp(u, lower_limit, upper_limit)

        # Any control inputs that are zero indicate a deadlock, so we should
        # switch to the exploratory mode and save the hit points for any rows that
        # are switching
        # switch_to_exploratory = (u.norm(dim=-1) <= 0.2).reshape(-1)
        switch_to_exploratory = (V_nexts > (1 - self.V_lambda) * V)[
            best_option_idx
        ].reshape(-1)
        switch_to_exploratory.logical_and_(
            self.controller_mode == self.GOAL_SEEKING_MODE
        )
        self.hit_points_V[switch_to_exploratory] = V.view(batch_size, num_options, -1)[
            switch_to_exploratory, 0
        ].detach()
        self.hit_points_h[switch_to_exploratory] = h.view(batch_size, num_options, -1)[
            switch_to_exploratory, 0
        ].detach()
        self.num_exploratory_steps[switch_to_exploratory] = 0.0

        # Do nothing for any points that have reached the goal (measured by V)
        goal_reached = x[:, :2].norm(dim=-1) < 0.1
        u[goal_reached] *= 0.0
        switch_to_exploratory[goal_reached] = False

        return u, best_option_cost.type_as(u), switch_to_exploratory

    def u_exploratory(
        self,
        x: torch.Tensor,
        o: torch.Tensor,
        h: torch.Tensor,
        V: torch.Tensor,
        idxs: torch.Tensor,
        u_options: torch.Tensor,
        x_nexts: torch.Tensor,
        o_nexts: torch.Tensor,
        h_nexts: torch.Tensor,
        V_nexts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the control input for the observations o and state x in the
        exploratory mode.

        args:
            x: bs x self.dynamics_model.n_dims tensor of states
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
            h: bs x 1 tensor of barrier function values
            V: bs x 1 tensor of lyapunov function values
            idxs: bs * N x 2 tensor of indices into *_nexts and u_options
            u_options: N x n_controls tensor of possible control inputs, where
                       N = self.lookahead_grid_n ^ n_controls (same below)
            x_nexts: bs * N x self.dynamics_model.n_dims tensor of next states
            o_nexts: bs * N x self.dynamics_model.obs_dim x self.dynamics_model.n_obs
                     tensor of observations at the next state
            h_nexts: bs * N x 1 tensor of barrier function values at the next state
            V_nexts: bs * N x 1 tensor of Lyapunov function values at the next state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control values
            cost: bs tensor of cost for each control action
            switch_to_goal_seeking: bs tensor of booleans indicating which rows should
                                    switch controller modes to goal seeking
        """
        # The exploration controller is a one-step lookahead controller that executes
        # a random walk by varying the previous control in a way that *on average*
        # causes the Lyapunov function to decrease but introduces sufficient variance
        # to escape any local minimum. We do this through reverse simulated annealing,
        # where we start with zero temperature (which induces similar behavior to the
        # goal-seeking mode), and gradually increasing temperature until the local
        # minimum is escaped (signalled by the Lyapunov function decreasing below its
        # value at the hit point)
        #
        # We do this by discretizing the action space and searching over the resulting
        # grid. We use approximate lookahead dynamics to propagate the provided
        # observations forward one step without querying the geometry model.
        #
        # The energy function is similar to the cost used in the goal-seeking mode, with
        # an extra term to encourage staying near the hit contour of the barrier
        # function
        #
        #       Q * (V_{t+1} - (1 - lambda) V_t) + R ||u - u_prev||^2
        #       + P * [h_0 - h_{t+1} - eps_h]_+
        # s.t.
        #
        #       h_{t+1} - (1 - alpha) h_t <= 0
        #
        # where the constraint is relaxed to a cost with a large penalty
        Q = 0.0
        R = 0.1
        P = 1000.0
        eps_h = 0.5

        # We now want to track which element of the search grid is best for each
        # row of the batched input. Create a tensor of costs for each option in each
        # batch.
        batch_size = x.shape[0]
        num_options = u_options.shape[0]
        costs = torch.zeros(batch_size * num_options, 1)

        # Add the barrier constraint penalty
        costs.add_(
            self.lookahead_dual_penalty
            * F.leaky_relu(h_nexts - (1 - self.h_alpha) * h, negative_slope=0.001)
        )
        # Add a cost to try to stay near the contour of the barrier function
        costs.add_(
            P
            * F.leaky_relu(
                (h_nexts - self.hit_points_h[idxs[:, 0]]).abs() - eps_h,
                negative_slope=0.01,
            )
        )
        # Add a cost to encourage Lyapunov decrease when possible
        costs.add_(Q * (V_nexts - (1 - self.V_lambda) * V))

        # Also impose a cost encouraging forward motion for turtlebot
        costs.add_(-R * (u_options[idxs[:, 1], 0] ** 2).unsqueeze(-1))

        # If we're debugging, loop through the options and visualize them
        if (
            self.debug_mode_exploratory
            and torch.allclose(
                self.controller_mode, torch.ones_like(self.controller_mode)
            )
            and (h > 0).any()
        ):
            for idx, idx_tensor in enumerate(idxs):
                state_idx, option_idx = idx_tensor
                # Reshape the control option to match the batch size
                u_option = u_options[option_idx, :].reshape(1, -1)  # 1 x n_controls

                # Get the cost for this option
                u_cost = -(u_option[:, 0] ** 2).sum(dim=-1)

                # Get the next state and observation from lookahead
                x_next = x_nexts[idx, :].unsqueeze(0)
                o_next = o_nexts[idx, :, :].unsqueeze(0)
                V_next = V_nexts[idx, :].unsqueeze(0)
                h_next = h_nexts[idx, :].unsqueeze(0)

                # Get the violation of the barrier decrease constraint
                barrier_function_violation = F.leaky_relu(
                    h_next - (1 - self.h_alpha) * h, negative_slope=0.001
                ).reshape(-1)
                # Also try to stay near the contour of the barrier function
                bf_tracking = (h_next - self.hit_points_h).abs()
                bf_tracking = F.leaky_relu(bf_tracking - eps_h, negative_slope=0.01)
                bf_tracking = bf_tracking.reshape(-1)
                # and get the LF decrease
                lyapunov_function_change = V_next - (1 - self.V_lambda) * V
                lyapunov_function_change = lyapunov_function_change.reshape(-1)

                print("=============")
                print(f"x: {x}")
                print(f"h: {h}")
                print(f"V: {V}")
                print(f"u_option: {u_option}")
                print(f"x_next: {x_next}")
                print(f"h_next: {h_next}")
                print(f"V_next: {V_next}")
                print(f"bf violation = {barrier_function_violation}")
                print(f"lf change = {lyapunov_function_change}")
                print(f"cost = {costs[idx]}")

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

        # Now randomly select an option for each row based on cost (= energy in the
        # parlance of simulated annealing). Convert costs to probabilities of selection
        # according to P = exp(-cost) (normalized)
        selection_probabilities = -costs.view(batch_size, num_options, -1).squeeze()
        selection_probabilities = F.softmax(selection_probabilities, dim=-1)
        chosen_option_idx = torch.multinomial(selection_probabilities, 1).reshape(-1)
        _, chosen_option_idx = torch.max(selection_probabilities, dim=-1)

        # Extract the control and cost for this option
        u = u_options[chosen_option_idx].reshape(-1, 1, u_options.shape[1])
        u_cost = costs[torch.arange(batch_size) * num_options + chosen_option_idx, :]

        if self.debug_mode_exploratory and torch.allclose(
            self.controller_mode, torch.ones_like(self.controller_mode)
        ):
            print("costs")
            print(costs)
            print("selection_probabilities")
            print(selection_probabilities)
            print(f"chose option {chosen_option_idx}, u = {u}")

        # Clamp to make sure we don't violate any control limits
        upper_limit, lower_limit = self.dynamics_model.control_limits
        u = torch.clamp(u, lower_limit, upper_limit)

        # Once in the exploratory mode, we can transition back to the goal seeking
        # mode once the Lyapunov function value has decreased below the hit point value
        V_next = V_nexts[torch.arange(batch_size) * num_options + chosen_option_idx, :]

        switch_to_goal_seeking = (
            V_next < (1 - self.V_lambda) * self.hit_points_V
        ).reshape(-1)
        switch_to_goal_seeking.logical_and_(
            self.controller_mode == self.EXPLORATORY_MODE
        )

        return u, u_cost.type_as(u), switch_to_goal_seeking

    def u_from_obs(self, x: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """Returns the control input at a given state. Computes the observations and
        barrier function value at this state before computing the control.
        """
        h = self.h(x, o)
        V = self.V(x)
        u, _ = self.u_(x, o, h, V)
        return u

    def u(self, x: torch.Tensor, reset: bool = False) -> torch.Tensor:
        """Returns the control input at a given state. Computes the observations and
        barrier function value at this state before computing the control.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            reset: if True, reset the modes of the controller
        """
        if reset:
            self.reset_controller(x)

        obs = self.get_observations(x)
        return self.u_from_obs(x, obs)

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
        eps = 1e-2
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
        eps = 1e-1

        # Get the barrier and lyapunov function at this current state
        h_t = self.h(x, o)
        V_t = self.V(x)

        # Get the control input
        self.reset_controller(x)
        u_t, u_cost = self.u_(x, o, h_t, V_t)

        # Penalize the cost
        barrier_loss = 1e0 * F.relu(eps + u_cost)[torch.logical_not(unsafe_mask)].mean()
        loss.append(("Barrier descent loss", barrier_loss))

        return loss

    def tuning_loss(
        self,
        x: torch.Tensor,
        o: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate a loss that tunes the BF and LF into well-defined shapes

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

        # Get the barrier and lyapunov function at this current state
        h_t = self.h(x, o)
        V_t = self.V(x)

        # Make h act like negative distance with some buffer
        min_dist, _ = o.norm(dim=1).min(dim=-1)
        min_dist = min_dist.reshape(-1, 1)
        h_tuning_distance = 0.2 - min_dist
        h_tuning_loss = 1e0 * ((h_t - h_tuning_distance) ** 2).sum(dim=-1).mean()
        loss.append(("H tuning loss", h_tuning_loss))

        # Make V act like a norm measuring range and angle from the origin
        distance_squared = (x[:, :2] ** 2).sum(dim=-1).reshape(-1, 1)
        # Phi is the angle from the current heading towards the origin
        angle_from_bot_to_origin = torch.atan2(-x[:, 1], -x[:, 0])
        theta = x[:, 2]
        phi = angle_from_bot_to_origin - theta
        # First, wrap the angle error into [-pi, pi]
        phi = torch.atan2(torch.sin(phi), torch.cos(phi))
        phi = phi.reshape(-1, 1)
        V_tuning = 1.0 * distance_squared + 0.5 * (1 - torch.cos(phi))
        V_tuning_loss = 1e0 * ((V_t - V_tuning) ** 2).sum(dim=-1).mean()
        loss.append(("V tuning loss", V_tuning_loss))

        return loss

    def losses(self):
        """Return a list of loss functions"""
        return [self.boundary_loss, self.descent_loss, self.tuning_loss]

    def accuracies(self):
        """Return a list of loss+accuracy functions"""
        return [
            self.boundary_loss,
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
        # Reset this controller then return the simulation results
        self.reset_controller(x_init)

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
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.primal_learning_rate,
        )

        self.opt_idx_dict = {0: "all"}

        return [opt]
