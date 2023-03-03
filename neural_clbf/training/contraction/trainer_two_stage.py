"""A Python class for training the contraction metric and controller networks"""
from collections import OrderedDict
from itertools import product
from typing import cast, Callable, Dict, List, Tuple, Union
import subprocess
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from simulation import (
    simulate,
    generate_random_reference,
    DynamicsCallable,
)

from nonlinear_mpc_controller import turtlebot_mpc_casadi_torch  # noqa
from trainer import Trainer


class TwoStageTrainer(Trainer):
    """
    Run batches of training in between searching for counterexamples.

    Learns the metric first and then the control policy.
    """

    def initialize_data(self):
        # Generate data using self.n_trajs trajectories of length batch_size
        T = self.batch_size * self.controller_dt + self.expert_horizon
        print("Constructing initial dataset...")
        # get these trajectories from a larger range of errors than we expect in testing
        error_bounds_demonstrations = [1.5 * bound for bound in self.error_bounds]
        x_init, x_ref, u_ref = generate_random_reference(
            self.n_trajs,
            T,
            self.controller_dt,
            self.n_state_dims,
            self.n_control_dims,
            self.state_space,
            self.control_bounds,
            error_bounds_demonstrations,
            self.dynamics,
        )
        traj_length = x_ref.shape[1]

        # Create some places to store the simulation results
        x = torch.zeros((self.n_trajs, traj_length, self.n_state_dims))
        x_dot = torch.zeros((self.n_trajs, traj_length, self.n_state_dims))
        x[:, 0, :] = x_init
        u_expert = torch.zeros((self.n_trajs, traj_length, self.n_control_dims))
        u_current = torch.zeros((self.n_control_dims,))

        # The expert policy requires a sliding window over the trajectory, so we need
        # to iterate through that trajectory.
        # Make sure we don't overrun the end of the reference while planning
        n_steps = traj_length - int(self.expert_horizon / self.controller_dt)
        dynamics_updates_per_control_update = int(self.controller_dt / self.sim_dt)
        for traj_idx in tqdm(range(self.n_trajs)):
            traj_range = range(n_steps - 1)
            for tstep in traj_range:
                # Get the current states and references
                x_current = x[traj_idx, tstep].reshape(-1, self.n_state_dims).clone()

                # Pick out sliding window into references for use with the expert
                x_ref_expert = (
                    x_ref[
                        traj_idx,
                        tstep : tstep + int(self.expert_horizon // self.controller_dt),
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )
                u_ref_expert = (
                    u_ref[
                        traj_idx,
                        tstep : tstep + int(self.expert_horizon // self.controller_dt),
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )

                # Run the expert
                u_current = torch.tensor(
                    self.expert_controller(
                        x_current.detach().cpu().numpy().squeeze(),
                        x_ref_expert,
                        u_ref_expert,
                    )
                )

                u_expert[traj_idx, tstep, :] = u_current

                # Add a bit of noise to widen the distribution of states
                u_current += torch.normal(
                    0, self.demonstration_noise * torch.tensor(self.control_bounds)
                )

                # Update state
                for i in range(dynamics_updates_per_control_update):
                    dx = self.dynamics(
                        x_current,
                        u_current.reshape(-1, self.n_control_dims),
                    )
                    x_current += self.sim_dt * dx
                    if i == 0:
                        x_dot[traj_idx, tstep, :] = dx
                x[traj_idx, tstep + 1, :] = x_current

            # plt.plot(x[traj_idx, :n_steps, 0], x[traj_idx, :n_steps, 1], "-")
            # plt.plot(x_ref[traj_idx, :n_steps, 0], x_ref[traj_idx, :n_steps, 1], ":")
            # plt.plot(x[traj_idx, 0, 0], x[traj_idx, 0, 1], "ko")
            # plt.plot(x_ref[traj_idx, 0, 0], x_ref[traj_idx, 0, 1], "ko")
            # plt.show()

            # plt.plot(u_expert[traj_idx, :, 0], "r:")
            # plt.plot(u_expert[traj_idx, :, 1], "r--")
            # plt.plot(u_ref[traj_idx, :, 0], "k:")
            # plt.plot(u_ref[traj_idx, :, 1], "k--")
            # plt.show()

        print(" Done!")

        # Reshape
        x_dot = x_dot[:, : tstep + 1, :].reshape(-1, self.n_state_dims)
        x = x[:, : tstep + 1, :].reshape(-1, self.n_state_dims)
        x_ref = x_ref[:, : tstep + 1, :].reshape(-1, self.n_state_dims)
        u_ref = u_ref[:, : tstep + 1, :].reshape(-1, self.n_control_dims)
        u_expert = u_expert[:, : tstep + 1, :].reshape(-1, self.n_control_dims)

        # Split data into training and validation and save it
        random_indices = torch.randperm(x.shape[0])
        val_points = int(x.shape[0] * self.validation_split)
        validation_indices = random_indices[:val_points]
        training_indices = random_indices[val_points:]

        self.x_ref_training = x_ref[training_indices]
        self.x_ref_validation = x_ref[validation_indices]

        self.u_ref_training = u_ref[training_indices]
        self.u_ref_validation = u_ref[validation_indices]

        self.x_training = x[training_indices]
        self.x_validation = x[validation_indices]

        self.x_dot_training = x_dot[training_indices]
        self.x_dot_validation = x_dot[validation_indices]

        self.u_expert_training = u_expert[training_indices]
        self.u_expert_validation = u_expert[validation_indices]

    def compute_pretrain_losses(
        self,
        x: torch.Tensor,
        x_dot: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the pre-training loss

        args:
            x - (batch_size + self.expert_horizon // self.controller_dt) x
                self.n_state_dims tensor of state
        """
        losses = {}
        losses["conditioning"] = self.contraction_loss_conditioning(x, None, None)
        losses["M"] = self.pretrain_contraction_loss_M(x, x_dot)
        return losses
    
    @torch.enable_grad()
    def pretrain_contraction_loss_M(
        self, 
        x: torch.Tensor,
        x_dot: torch.Tensor,
    ) -> torch.Tensor:
        
        x = x.requires_grad_()
        x_dot = x_dot.requires_grad_(False)
        M = self.M(x)
        Mdot = self.weighted_gradients(M, x_dot, x, detach=False)
        contraction_cond = Mdot + 2 * self.lambda_M * M

        loss = torch.tensor(0.0)
        loss += self.positive_definite_loss(-contraction_cond, eps=0.1)
        return loss

    def run_training(
        self,
        n_pretrain_steps: int,
        n_steps: int,
        debug: bool = False,
        finetune_M: bool = False,
        sim_every_n_steps: int = 1,
    ):
        """
        Train for n_steps

        args:
            n_steps: the number of steps to train for
            debug: if True, log losses to tensorboard and display a progress bar
            sim_every_n_steps: run a simulation and save it to tensorboard every n steps

        returns:
            A pair of lists, each of which contains tuples defining the metric and
            policy network, respectively. Each tuple represents a layer, either:

                ("linear", numpy matrix of weights, numpy matrix of biases)
                ("relu", empty matrix, empty matrix)

            as well as two additional lists of training and test loss
        """
        # Log histories of training and test loss
        training_losses = []
        test_losses = []

        # Find out how many training and test examples we have
        N_train = self.x_training.shape[0]
        N_test = self.x_validation.shape[0]
        
        pretrain_epochs = range(n_pretrain_steps)
        for pretrain_epoch in pretrain_epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {pretrain_epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_dot = self.x_dot_training[batch_indices]

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute contraction metric loss and backpropagate
                losses = {}
                losses = self.compute_pretrain_losses(x, x_dot)
                
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_accumulated += losses["M"].detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm/pretrain", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/pretrain", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/pretrain",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/pretrain",
                M_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            # TODO: Is there a way to visualize the regions 
            # where the contraction metric is valid / invalid respectively?

            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {pretrain_epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_dot = self.x_dot_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_pretrain_losses(x, x_dot)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/pretrain_test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Reset global steps for training
        self.global_steps = 0

        # Freeze learned metric if so configured
        self.A.requires_grad_(finetune_M)

        epochs = range(n_steps)
        for epoch in epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_policy_accumulated = 0.0
            M_loss_intrinsic_accumulated = 0.0
            u_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_dot = self.x_dot_training[batch_indices]
                x_ref = self.x_ref_training[batch_indices]
                u_ref = self.u_ref_training[batch_indices]
                u_expert = self.u_expert_training[batch_indices]

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss and backpropagate
                M_loss_intrinsic = self.pretrain_contraction_loss_M(x, x_dot)
                losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()
                loss_accumulated += M_loss_intrinsic.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_policy_accumulated += losses["M"].detach().item()
                if "u" in losses:
                    u_loss_accumulated += losses["u"].detach().item()
                M_loss_intrinsic_accumulated += M_loss_intrinsic.detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm/train", total_norm, self.global_steps)
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm/train", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/train", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/train",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train_intrinsic",
                M_loss_intrinsic_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train_policy",
                M_loss_policy_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "u Loss/train",
                u_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            # Also log a simulation plot every so often
            if sim_every_n_steps == 1 or epoch % sim_every_n_steps == 1:
                # Generate a random reference trajectory
                N_batch = 1  # number of test trajectories
                T = 20.0  # length of trajectory
                x_init, x_ref_sim, u_ref_sim = generate_random_reference(
                    N_batch,
                    T,
                    self.sim_dt,
                    self.n_state_dims,
                    self.n_control_dims,
                    self.state_space,
                    self.control_bounds,
                    self.error_bounds,
                    self.dynamics,
                )

                # Simulate
                x_sim, u_sim, M_sim, dMdt_sim = simulate(
                    x_init,
                    x_ref_sim,
                    u_ref_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.u,
                    self.metric_value,
                    self.metric_derivative_t,
                    self.control_bounds,
                )
                x_sim = x_sim.detach()
                u_sim = u_sim.detach()
                M_sim = M_sim.detach()
                dMdt_sim = dMdt_sim.detach()

                # Make a plot for state error
                t_range = np.arange(0, T, self.sim_dt)
                fig, ax = plt.subplots()
                fig.set_size_inches(8, 4)

                # Plot the reference and actual trajectories
                ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
                ax.plot(
                    t_range,
                    (x_ref_sim - x_sim).norm(dim=-1).cpu().detach().numpy().squeeze(),
                    linestyle=":",
                )
                ax.set_xlabel("time (s)")
                ax.set_ylabel("State Error")

                # Save the figure
                self.writer.add_figure(
                    "Simulated State Trajectory/Error",
                    fig,
                    self.global_steps,
                )

                # Make a plot for each control
                for control_idx in range(self.n_control_dims):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(8, 4)

                    # Plot the reference and actual trajectories
                    ax.plot([], [], linestyle=":", color="k", label="Reference")
                    ax.plot(
                        t_range[1:],
                        u_ref_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                        linestyle=":",
                    )
                    ax.set_prop_cycle(None)  # Re-use colors for the reference
                    ax.plot([], [], linestyle="-", color="k", label="Actual")
                    ax.plot(
                        t_range[1:],
                        u_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                        linestyle="-",
                    )
                    ax.set_xlabel("time (s)")
                    ax.set_ylabel(f"Control {control_idx}")
                    ax.legend()

                    # Save the figure
                    self.writer.add_figure(
                        f"Simulated Control Trajectory/Control {control_idx}",
                        fig,
                        self.global_steps,
                    )

                # Also make a phase plane plot
                fig, ax = plt.subplots()
                fig.set_size_inches(8, 8)

                # Plot the reference and actual trajectories
                ax.plot([], [], linestyle=":", color="k", label="Reference")
                ax.plot([], [], marker="o", color="k", label="Start")
                ax.plot(
                    x_ref_sim[:, :, 0].T.cpu().detach().numpy(),
                    x_ref_sim[:, :, 1].T.cpu().detach().numpy(),
                    linestyle=":",
                )
                ax.plot(
                    x_ref_sim[:, 0, 0].T.cpu().detach().numpy(),
                    x_ref_sim[:, 0, 1].T.cpu().detach().numpy(),
                    marker="o",
                    color="k",
                )
                ax.set_prop_cycle(None)  # Re-use colors for the reference
                ax.plot([], [], linestyle="-", color="k", label="Actual")
                ax.plot(
                    x_sim[:, :, 0].T.cpu().detach().numpy(),
                    x_sim[:, :, 1].T.cpu().detach().numpy(),
                    linestyle="-",
                )
                ax.plot(
                    x_sim[:, 0, 0].T.cpu().detach().numpy(),
                    x_sim[:, 0, 1].T.cpu().detach().numpy(),
                    marker="o",
                    color="k",
                )
                ax.legend()

                # Save the figure
                self.writer.add_figure(
                    "Phase Plane",
                    fig,
                    self.global_steps,
                )

                # Also plot the metric
                fig, ax = plt.subplots()
                fig.set_size_inches(8, 4)
                ax.plot([], [], linestyle="-", color="k", label="Metric")
                ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
                ax.plot(
                    t_range[:-1],
                    M_sim[:, :-1, 0].T.cpu().detach().numpy(),
                    linestyle="-",
                )
                # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
                # ax.plot(
                #     t_range[:-1],
                #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
                #     linestyle=":",
                # )
                # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
                # ax.plot(
                #     t_range[:-2],
                #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
                #     linestyle="--",
                # )
                ax.set_xlabel("time (s)")
                ax.legend()

                # Save the figure
                self.writer.add_figure(
                    "Simulated Metric",
                    fig,
                    self.global_steps,
                )

            # self.writer.close()

            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_ref = self.x_ref_validation[indices]
                    u_ref = self.u_ref_validation[indices]
                    u_expert = self.u_expert_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Create the list representations of the metric and policy network

        # The metric network starts with a normalization layer and ends with map from A
        # to M, which we have to manually add
        contraction_network_list = []
        contraction_network_list.append(
            (
                "linear",
                self.state_normalization_weights.detach().cpu().numpy(),
                self.state_normalization_bias.detach().cpu().numpy(),
            )
        )
        for layer_name, layer in self.metric_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                contraction_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                contraction_network_list.append(
                    ("relu", torch.Tensor(), torch.Tensor())
                )
        contraction_network_list.append(
            (
                "linear",
                self.A_to_M.detach().cpu().numpy(),
                torch.zeros(self.n_state_dims ** 2).detach().cpu().numpy(),
            )
        )

        policy_network_list = []
        # From a learning perspective, it helps to have x and (x-x_ref) as inputs to
        # the control policy; however, from a verification perspective it helps to
        # have x and x_ref as inputs. To get around this, we'll put a linear layer in
        # front of the policy network that takes [x; x_ref] to [x, x - x_ref]
        subtraction_weights = torch.block_diag(
            torch.eye(self.n_state_dims), -torch.eye(self.n_state_dims)
        )
        subtraction_weights[self.n_state_dims :, : self.n_state_dims] = torch.eye(
            self.n_state_dims
        )
        policy_network_list.append(
            (
                "linear",
                subtraction_weights.detach().cpu().numpy(),
                torch.zeros(2 * self.n_state_dims).detach().cpu().numpy(),
            )
        )
        # The policy network includes a normalization layer applied to both x
        # and x_error, so we need to construct a larger normalization matrix here
        policy_network_list.append(
            (
                "linear",
                torch.block_diag(
                    self.state_normalization_weights, self.error_normalization_weights
                )
                .detach()
                .cpu()
                .numpy(),
                torch.cat(
                    [self.state_normalization_bias, torch.zeros(self.n_state_dims)]
                )
                .detach()
                .cpu()
                .numpy(),
            )
        )
        for layer_name, layer in self.policy_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                policy_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                policy_network_list.append(("relu", np.array([]), np.array([])))
        # And include output normalization
        policy_network_list.append(
            (
                "linear",
                self.control_normalization_weights.detach().cpu().numpy(),
                torch.zeros(self.n_control_dims).detach().cpu().numpy(),
            )
        )

        # Return the network lists and the loss lists
        return (
            contraction_network_list,
            policy_network_list,
            training_losses,
            test_losses,
        )