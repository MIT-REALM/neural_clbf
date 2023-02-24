"""Simulate a rollout and plot in state space"""
import random
import time
from typing import cast, List, Tuple, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import torch
import tqdm

from neural_clbf.experiments import Experiment
from neural_clbf.systems.utils import ScenarioList

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller, NeuralObsBFController  # noqa
    from neural_clbf.systems import ObservableSystem  # noqa


class RolloutStateSpaceExperiment(Experiment):
    """An experiment for plotting rollout performance of controllers.

    Plots trajectories projected onto a 2D plane.
    """

    def __init__(
        self,
        name: str,
        start_x: torch.Tensor,
        plot_x_index: int,
        plot_x_label: str,
        plot_y_index: int,
        plot_y_label: str,
        other_index: Optional[List[int]] = None,
        other_label: Optional[List[str]] = None,
        scenarios: Optional[ScenarioList] = None,
        n_sims_per_start: int = 5,
        t_sim: float = 5.0,
    ):
        """Initialize an experiment for simulating controller performance.

        args:
            name: the name of this experiment
            plot_x_index: the index of the state dimension to plot on the x axis,
            plot_x_label: the label of the state dimension to plot on the x axis,
            plot_y_index: the index of the state dimension to plot on the y axis,
            plot_y_label: the label of the state dimension to plot on the y axis,
            other_index: the indices of the state dimensions to save
            other_label: the labels of the state dimensions to save
            scenarios: a list of parameter scenarios to sample from. If None, use the
                       nominal parameters of the controller's dynamical system
            n_sims_per_start: the number of simulations to run (with random parameters),
                              per row in start_x
            t_sim: the amount of time to simulate for
        """
        super(RolloutStateSpaceExperiment, self).__init__(name)

        # Save parameters
        self.start_x = start_x
        self.plot_x_index = plot_x_index
        self.plot_x_label = plot_x_label
        self.plot_y_index = plot_y_index
        self.plot_y_label = plot_y_label
        self.scenarios = scenarios
        self.n_sims_per_start = n_sims_per_start
        self.t_sim = t_sim
        self.other_index = [] if other_index is None else other_index
        self.other_label = [] if other_label is None else other_label

    @torch.no_grad()
    def run(self, controller_under_test: "Controller") -> pd.DataFrame:
        """
        Run the experiment, likely by evaluating the controller, but the experiment
        has freedom to call other functions of the controller as necessary (if these
        functions are not supported by all controllers, then experiments will be
        responsible for checking compatibility with the provided controller)

        args:
            controller_under_test: the controller with which to run the experiment
        returns:
            a pandas DataFrame containing the results of the experiment, in tidy data
            format (i.e. each row should correspond to a single observation from the
            experiment).
        """
        # Deal with optional parameters
        if self.scenarios is None:
            scenarios = [controller_under_test.dynamics_model.nominal_params]
        else:
            scenarios = self.scenarios
        
        if not hasattr(self, "other_index"):
            self.other_index = []
            self.other_label = []

        # Set up a dataframe to store the results
        results = []

        # Compute the number of simulations to run
        n_sims = self.n_sims_per_start * self.start_x.shape[0]

        # Determine the parameter range to sample from
        parameter_ranges = {}
        for param_name in scenarios[0].keys():
            param_max = max([s[param_name] for s in scenarios])
            param_min = min([s[param_name] for s in scenarios])
            parameter_ranges[param_name] = (param_min, param_max)

        # Generate a tensor of start states
        n_dims = controller_under_test.dynamics_model.n_dims
        n_controls = controller_under_test.dynamics_model.n_controls
        x_sim_start = torch.zeros(n_sims, n_dims).type_as(self.start_x)
        for i in range(0, self.start_x.shape[0]):
            for j in range(0, self.n_sims_per_start):
                x_sim_start[i * self.n_sims_per_start + j, :] = self.start_x[i, :]

        # Generate a random scenario for each rollout from the given scenarios
        random_scenarios = []
        for i in range(n_sims):
            random_scenario = {}
            for param_name in scenarios[0].keys():
                param_min = parameter_ranges[param_name][0]
                param_max = parameter_ranges[param_name][1]
                random_scenario[param_name] = random.uniform(param_min, param_max)
            random_scenarios.append(random_scenario)

        # Make sure everything's on the right device
        device = "cpu"
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device  # type: ignore
        x_current = x_sim_start.to(device)

        # Reset the controller if necessary
        if hasattr(controller_under_test, "reset_controller"):
            controller_under_test.reset_controller(x_current)  # type: ignore

        # See how long controller took
        controller_calls = 0
        controller_time = 0.0

        # Simulate!
        delta_t = controller_under_test.dynamics_model.dt
        num_timesteps = int(self.t_sim // delta_t)
        u_current = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
        controller_update_freq = int(controller_under_test.controller_period / delta_t)
        prog_bar_range = tqdm.trange(
            0, num_timesteps, desc="Controller Rollout", leave=True
        )
        for tstep in prog_bar_range:
            # Get the control input at the current state if it's time
            if tstep % controller_update_freq == 0:
                start_time = time.time()
                u_current = controller_under_test.u(x_current)
                end_time = time.time()
                controller_calls += 1
                controller_time += end_time - start_time

            # Get the barrier function if applicable
            h: Optional[torch.Tensor] = None
            if hasattr(controller_under_test, "h") and hasattr(
                controller_under_test.dynamics_model, "get_observations"
            ):
                controller_under_test = cast(
                    "NeuralObsBFController", controller_under_test
                )
                dynamics_model = cast(
                    "ObservableSystem", controller_under_test.dynamics_model
                )
                obs = dynamics_model.get_observations(x_current)
                h = controller_under_test.h(x_current, obs)

            # Get the Lyapunov function if applicable
            V: Optional[torch.Tensor] = None
            if hasattr(controller_under_test, "V") and h is None:
                V = controller_under_test.V(x_current)  # type: ignore

            # Log the current state and control for each simulation
            for sim_index in range(n_sims):
                log_packet = {"t": tstep * delta_t, "Simulation": str(sim_index)}

                # Include the parameters
                param_string = ""
                for param_name, param_value in random_scenarios[sim_index].items():
                    param_value_string = "{:.3g}".format(param_value)
                    param_string += f"{param_name} = {param_value_string}, "
                    log_packet[param_name] = param_value
                log_packet["Parameters"] = param_string[:-2]

                # Pick out the states to log and save them
                x_value = x_current[sim_index, self.plot_x_index].cpu().numpy().item()
                y_value = x_current[sim_index, self.plot_y_index].cpu().numpy().item()
                log_packet[self.plot_x_label] = x_value
                log_packet[self.plot_y_label] = y_value
                log_packet["state"] = x_current[sim_index, :].cpu().detach().numpy()

                for i, save_index in enumerate(self.other_index):
                    value = x_current[sim_index, save_index].cpu().numpy().item()
                    log_packet[self.other_label[i]] = value

                # Log the barrier function if applicable
                if h is not None:
                    log_packet["h"] = h[sim_index].cpu().numpy().item()
                # Log the Lyapunov function if applicable
                if V is not None:
                    log_packet["V"] = V[sim_index].cpu().numpy().item()

                results.append(log_packet)

            # Simulate forward using the dynamics
            for i in range(n_sims):
                xdot = controller_under_test.dynamics_model.closed_loop_dynamics(
                    x_current[i, :].unsqueeze(0),
                    u_current[i, :].unsqueeze(0),
                    random_scenarios[i],
                )
                x_current[i, :] = x_current[i, :] + delta_t * xdot.squeeze()

        return pd.DataFrame(results)

    def plot(
        self,
        controller_under_test: "Controller",
        results_df: pd.DataFrame,
        display_plots: bool = False,
    ) -> List[Tuple[str, figure]]:
        """
        Plot the results, and return the plot handles. Optionally
        display the plots.

        args:
            controller_under_test: the controller with which to run the experiment
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """

        # Set the color scheme
        sns.set_theme(context="talk", style="white")

        # Figure out how many plots we need (one for the rollout, one for h if logged,
        # and one for V if logged)
        num_plots = 1
        if "h" in results_df:
            num_plots += 1
        if "V" in results_df:
            num_plots += 1

        # Plot the state trajectories
        fig, ax = plt.subplots(1, num_plots)
        fig.set_size_inches(9 * num_plots, 6)

        # Assign plots to axes
        if num_plots == 1:
            rollout_ax = ax
        else:
            rollout_ax = ax[0]

        if "h" in results_df:
            h_ax = ax[1]
        if "V" in results_df:
            V_ax = ax[num_plots - 1]

        # Plot the rollout
        # sns.lineplot(
        #     ax=rollout_ax,
        #     x=self.plot_x_label,
        #     y=self.plot_y_label,
        #     style="Parameters",
        #     hue="Simulation",
        #     data=results_df,
        # )
        num_traces = len(results_df["Simulation"].unique())
        for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
            sim_mask = results_df["Simulation"] == sim_index
            rollout_ax.plot(
                results_df[sim_mask][self.plot_x_label].to_numpy(),
                results_df[sim_mask][self.plot_y_label].to_numpy(),
                linestyle="-",
                # marker="+",
                markersize=5,
                color=sns.color_palette(n_colors=num_traces)[plot_idx],
            )
            rollout_ax.set_xlabel(self.plot_x_label)
            rollout_ax.set_ylabel(self.plot_y_label)

        # Remove the legend -- too much clutter
        rollout_ax.legend([], [], frameon=False)

        # Plot the environment
        controller_under_test.dynamics_model.plot_environment(rollout_ax)

        # Plot the barrier function if applicable
        if "h" in results_df:
            # Get the derivatives for each simulation
            for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
                sim_mask = results_df["Simulation"] == sim_index

                h_ax.plot(
                    results_df[sim_mask]["t"].to_numpy(),
                    results_df[sim_mask]["h"].to_numpy(),
                    linestyle="-",
                    # marker="+",
                    markersize=5,
                    color=sns.color_palette(n_colors=num_traces)[plot_idx],
                )
                h_ax.set_ylabel("$h$")
                h_ax.set_xlabel("t")
                # Remove the legend -- too much clutter
                h_ax.legend([], [], frameon=False)

                # Plot a reference line at h = 0
                h_ax.plot([0, results_df["t"].max()], [0, 0], color="k")

                # Also plot the derivatives
                h_next = results_df[sim_mask]["h"][1:].to_numpy()
                h_now = results_df[sim_mask]["h"][:-1].to_numpy()
                alpha = controller_under_test.h_alpha  # type: ignore
                h_violation = h_next - (1 - alpha) * h_now

                h_ax.plot(
                    results_df[sim_mask]["t"][:-1].to_numpy(),
                    h_violation,
                    linestyle=":",
                    color=sns.color_palette(n_colors=num_traces)[plot_idx],
                )
                h_ax.set_ylabel("$h$ violation")

        # Plot the lyapunov function if applicable
        if "V" in results_df:
            for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
                sim_mask = results_df["Simulation"] == sim_index
                V_ax.plot(
                    results_df[sim_mask]["t"].to_numpy(),
                    results_df[sim_mask]["V"].to_numpy(),
                    linestyle="-",
                    # marker="+",
                    markersize=5,
                    color=sns.color_palette(n_colors=num_traces)[plot_idx],
                )
            # sns.lineplot(
            #     ax=V_ax,
            #     x="t",
            #     y="V",
            #     style="Parameters",
            #     hue="Simulation",
            #     data=results_df,
            # )
            V_ax.set_ylabel("$V$")
            V_ax.set_xlabel("t")
            # Remove the legend -- too much clutter
            V_ax.legend([], [], frameon=False)

            # Plot a reference line at V = 0
            V_ax.plot([0, results_df.t.max()], [0, 0], color="k")

        fig_handle = ("Rollout (state space)", fig)

        if display_plots:
            plt.show()
            return []
        else:
            return [fig_handle]
