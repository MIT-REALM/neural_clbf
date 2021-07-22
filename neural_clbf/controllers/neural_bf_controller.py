import itertools
from typing import Tuple, List, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from neural_clbf.systems import ObservableSystem
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
        observations -> encoder -> fully-connected layers -> h

    u:
        observations + h -> encoder -> fully-connected layers -> u

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
        u_hidden_layers: int = 2,
        u_hidden_size: int = 48,
        h_alpha: float = 0.9,
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-3,
        epochs_per_episode: Optional[int] = None,
        validation_dynamics_model: Optional[ObservableSystem] = None,
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
            u_hidden_layers: number of hidden layers to use for the policy network
            u_hidden_size: number of neurons per hidden layer in the policy network
            h_alpha: convergence rate for the BF
            controller_period: the timestep to use in simulating forward Vdot
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the BF decrease loss
            epochs_per_episode: optionally gather additional training data every few
                                epochs. If none, no new data will be gathered.f
            validation_dynamics_model: optionally provide a dynamics model to use during
                                       validation
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
        self.epochs_per_episode = epochs_per_episode

        # ----------------------------------------------------------------------------
        # Define the encoder network
        # ----------------------------------------------------------------------------
        self.input_size = self.dynamics_model.obs_dim
        self.encoder_hidden_layers = encoder_hidden_layers
        self.encoder_hidden_size = encoder_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.encoder_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.encoder_layers["input_linear"] = nn.Conv1d(
            self.input_size, self.encoder_hidden_size, 1  # kernel size = 1
        )
        self.encoder_layers["input_activation"] = nn.ReLU()
        for i in range(self.encoder_hidden_layers):
            self.encoder_layers[f"layer_{i}_linear"] = nn.Conv1d(
                self.encoder_hidden_size, self.encoder_hidden_size, 1  # kernel size = 1
            )
            self.encoder_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.encoder_nn = nn.Sequential(self.encoder_layers)

        # ----------------------------------------------------------------------------
        # Define the BF network, which we denote h
        # ----------------------------------------------------------------------------
        self.h_hidden_layers = h_hidden_layers
        self.h_hidden_size = h_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.h_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.h_layers["input_linear"] = nn.Linear(
            self.encoder_hidden_size, self.h_hidden_size
        )
        self.h_layers["input_activation"] = nn.ReLU()
        for i in range(self.h_hidden_layers):
            self.h_layers[f"layer_{i}_linear"] = nn.Linear(
                self.h_hidden_size, self.h_hidden_size
            )
            self.h_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.h_layers["output_linear"] = nn.Linear(self.h_hidden_size, 1)
        self.h_nn = nn.Sequential(self.h_layers)

        # ----------------------------------------------------------------------------
        # Define the policy network, which we denote u
        # ----------------------------------------------------------------------------
        self.u_hidden_layers = u_hidden_layers
        self.u_hidden_size = u_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.u_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.u_layers["input_linear"] = nn.Linear(
            self.encoder_hidden_size + 1,  # add one for the barrier function input
            self.u_hidden_size,
        )
        self.u_layers["input_activation"] = nn.ReLU()
        for i in range(self.u_hidden_layers):
            self.u_layers[f"layer_{i}_linear"] = nn.Linear(
                self.u_hidden_size, self.u_hidden_size
            )
            self.u_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.u_layers["output_linear"] = nn.Linear(
            self.u_hidden_size, self.dynamics_model.n_controls
        )
        self.u_nn = nn.Sequential(self.u_layers)

        # Associated with u, we also have the network that decides when to override
        # the nominal controller, based on a simple linear+sigmoid decision rule
        self.intervention_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.intervention_layers["linear"] = nn.Linear(1, 1)
        self.intervention_layers["activation"] = nn.Sigmoid()
        self.intervention_nn = nn.Sequential(self.intervention_layers)

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

    def encoder(self, o: torch.Tensor):
        """Encode the observations o to a fixed-size representation via a permutation-
        invariant encoding

        args:
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
        returns:
            e: bs x self.encoder_hidden_size encoding of the observations
        """
        # First run the observations through the encoder network, which uses
        # convolutional layers with kernel size 1 to implement a fully connected
        # network that doesn't care what the length of the last dimension of the input
        # is (the same transformation will be applied to each point).
        e = self.encoder_nn(o)

        # Then max-pool over the last dimension
        e, _ = e.max(dim=-1)

        return e

    def h(self, o: torch.Tensor):
        """Return the BF value for the observations o

        args:
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
        returns:
            h: bs tensor of BF values
        """
        # Encode the observations
        encoded_obs = self.encoder(o)

        # Then get the barrier function value
        h = self.h_nn(encoded_obs)

        return h

    def u_(self, x: torch.Tensor, o: torch.Tensor, h: torch.Tensor):
        """Return the control input for the observations o and state x.

        The state x is used only as an input to the nominal controller, not to the
        barrier function controller.

        args:
            x: bs x self.dynamics_model.n_dims tensor of states, used only for
               evaluating the nominal controller
            o: bs x self.dynamics_model.obs_dim x self.dynamics_model.n_obs tensor of
               observations
            h: bs x 1 tensor of barrier function values
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control values
        """
        # The architecture of this controller is as follows. h is used to determine
        # whether or not to supercede the nominal controller, and o is used to determine
        # the superceding control input.
        #
        # x -> nominal_controller ---> (*) --------------> (+) ---> u
        #                               ^                   ^
        # h -> linear ---> sigmoid ----/--> (1 - ) -> (*) -/
        #               \                             ^
        # o -> encoder --> u_nn ---------------------/

        # Get the nominal control
        u_nominal = self.dynamics_model.u_nominal(x)

        # Get the decision signal (from 0 to 1 due to sigmoid output)
        # decision = self.intervention_nn(h)
        decision = torch.sigmoid(20 * (h + 0.25))

        # Get the control input from the encoded observations and the barrier function
        # value
        encoded_obs = self.encoder(o)
        encoded_obs_w_h = torch.cat((encoded_obs, h), 1)
        u_learned = self.u_nn(encoded_obs_w_h)

        # Blend the learned control with the nominal control based on the decision
        # value
        u = (1 - decision) * u_nominal + decision * u_learned

        # Then clamp the control input based on the specified limits
        u_upper, u_lower = self.dynamics_model.control_limits
        u = torch.clamp(u, u_lower, u_upper)

        return u

    def u(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the control input at a given state. Computes the observations and
        barrier function value at this state before computing the control.
        """
        obs = self.get_observations(x)
        h = self.h(obs)
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
        eps = 1e-2
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        h = self.h(o)

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
        eps = 0.1

        # We'll encourage satisfying the BF conditions by...
        #
        #   1) Getting the change in the barrier function value after one control
        #      period has elapsed, and computing the violation of BF conditions
        #      based on that change.

        # Get the barrier function at this current state
        h_t = self.h(o)

        # Get the control input
        u_t = self.u_(x, o, h_t)

        # Propagate the dynamics forward via a zero-order hold for one control period
        x_tplus1 = self.dynamics_model.zero_order_hold(x, u_t, self.controller_period)

        # Get the barrier function at this new state
        o_tplus1 = self.get_observations(x_tplus1)
        h_tplus1 = self.h(o_tplus1)

        # The discrete-time barrier function is h(t+1) - h(t) \leq -alpha h(t)
        # which reformulate to h(t+1) - (1 - alpha) h(t) \leq 0
        # First, adjust the convergence rate according to the timestep
        adjusted_alpha = self.controller_period * self.h_alpha
        barrier_function_violation = h_tplus1 - (1 - adjusted_alpha) * h_t
        barrier_function_violation = F.relu(eps + barrier_function_violation)
        barrier_loss = 1e1 * barrier_function_violation.mean()
        barrier_acc = (barrier_function_violation <= eps).sum() / x.shape[0]

        loss.append(("Barrier descent loss", barrier_loss))

        if accuracy:
            loss.append(("Barrier descent accuracy", barrier_acc))

        return loss

    def controller_loss(
        self,
        x: torch.Tensor,
        o: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the controller due to normalization

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
        loss = []

        # Add a very small term encouraging control inputs that match the nominal
        h = self.h(o)
        u_t = self.u_(x, o, h)
        u_nominal = self.dynamics_model.u_nominal(x)
        u_norm = (u_t - u_nominal).norm()
        loss.append(("||u - u_nominal||", 1e-3 * u_norm))

        return loss

    def losses(self):
        """Return a list of loss functions"""
        return [
            self.boundary_loss,
            self.descent_loss,
            self.controller_loss,
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
        # We automatically run experiments every few epochs

        # Only plot every 5 epochs
        if self.current_epoch % 5 != 0:
            return

        self.experiment_suite.run_all_and_log_plots(
            self, self.logger, self.current_epoch
        )

        # Now swap in the validation dynamics model and run the experiments again
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
