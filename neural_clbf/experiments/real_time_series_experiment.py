"""A mock experiment for use in testing"""
from copy import copy
import random
from typing import List, Tuple, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import torch
import tqdm

from neural_clbf.experiments import Experiment
from neural_clbf.systems.utils import ScenarioList

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller  # noqa


class RealTimeSeriesExperiment(Experiment):
    """An experiment for plotting actual
    performance of controller on turtlebot.
    
    Plots trajectories as a function of time.
    """

    def __init__(
        self,
        name: str,
        start_x: torch.Tensor,
        plot_x_indices: List[int],
        plot_x_labels: List[str],
        plot_u_indices: List[int],
        plot_u_labels: List[str],
        scenarios: Optional[ScenarioList] = None,
        n_sims_per_start: int = 5,
        t_sim: float = 5.0,
    ):
        """Initialize an experiment for controller performance on turtlebot.
        args:
            name: the name of this experiment
            plot_x_indices: a list of the indices of the state variables to plot
            plot_x_labels: a list of the labels for each state variable trace
            plot_u_indices: a list of the indices of the control inputs to plot
            plot_u_labels: a list of the labels for each control trace
            scenarios: a list of parameter scenarios to sample from. If None, use the
                       nominal parameters of the controller's dynamical system
            n_sims_per_start: the number of simulations to run (with random parameters),
                              per row in start_x
            t_sim: the amount of time to simulate for
        """
        super(RealTimeSeriesExperiment, self).__init__(name)

        # Save parameters
        self.start_x = start_x
        self.plot_x_indices = plot_x_indices
        self.plot_x_labels = plot_x_labels
        self.plot_u_indices = plot_u_indices
        self.plot_u_labels = plot_u_labels
        self.scenarios = scenarios
        self.n_sims_per_start = n_sims_per_start
        self.t_sim = t_sim

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
        #TODO find out what exactly scenarios is; I think it's a starting point.
        # Make sure to either modify this to fit real experiment or make sure
        # it is picking a singular starting point
        # Deal with optional parameters
        if self.scenarios is None:
            scenarios = [controller_under_test.dynamics_model.nominal_params]
        else:
            scenarios = self.scenarios

        # Set up a dataframe to store the results
        results_df = pd.DataFrame()

        # Compute the number of simulations to run
        #TODO we're only going to run one "simulation" so
        # this probably will need to be changed
        n_sims = self.n_sims_per_start * self.start_x.shape[0]

        # Determine the parameter range to sample from
        parameter_ranges = {}
        for param_name in scenarios[0].keys():
            param_max = max([s[param_name] for s in scenarios])
            param_min = min([s[param_name] for s in scenarios])
            parameter_ranges[param_name] = (param_min, param_max)

        # Generate a tensor of start states
        #TODO this may need to be modified to make it work for only a 
        # single start point since we're only running one experiment
        n_dims = controller_under_test.dynamics_model.n_dims
        n_controls = controller_under_test.dynamics_model.n_controls
        x_sim_start = torch.zeros(n_sims, n_dims).type_as(self.start_x)
        for i in range(0, self.start_x.shape[0]):
            for j in range(0, self.n_sims_per_start):
                x_sim_start[i * self.n_sims_per_start + j, :] = self.start_x[i, :]

        # Generate a random scenario for each rollout from the given scenarios
        #TODO We don't want a random scenario, we want to run one scenario: the turtlebot
        # starts in one position and then moves to the origin
        random_scenarios = []
        for i in range(n_sims):
            random_scenario = {}
            for param_name in scenarios[0].keys():
                param_min = parameter_ranges[param_name][0]
                param_max = parameter_ranges[param_name][1]
                random_scenario[param_name] = random.uniform(param_min, param_max)
            random_scenarios.append(random_scenario)

        # Make sure everything's on the right device
        # this probably doesn't need to be modified
        device = "cpu"
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device  # type: ignore

        # Simulate!
        #TODO delta_t is probably fine, try to match it with turtlebot
        # update rate too if possible
        delta_t = controller_under_test.dynamics_model.dt
        
        #TODO probably should modify to make it run until we reach the
        # goal rather than until the simulation time is up
        num_timesteps = int(self.t_sim // delta_t)
        
        #TODO change this to get position and attitude from turtlebot
        x_current = x_sim_start.to(device)
        
        #TODO send this to the turtlebot as a command
        u_current = torch.zeros(x_sim_start.shape[0], n_controls, device=device)
        
        #TODO try to match controller update frequency with turtlebot update frequency
        controller_update_freq = int(controller_under_test.controller_period / delta_t)
        prog_bar_range = tqdm.trange(
            0, num_timesteps, desc="Controller Rollout", leave=True
        )
        
        #TODO again, might need to be modified if we're doing this based on when
        # the turtlebot reaches the goal rather than a set amount of simulation time
        for tstep in prog_bar_range:
            # Get the control input at the current state if it's time
            #TODO make sure x_current is coming from turtlebot
            if tstep % controller_update_freq == 0:
                u_current = controller_under_test.u(x_current)

            # Log the current state and control for each simulation
            #TODO make sure we're only doing one "simulation"
            for sim_index in range(n_sims):
                base_log_packet = {"t": tstep * delta_t}

                # Include the parameters
                #TODO how much of this do we need for the actual turtlebot?
                param_string = ""
                for param_name, param_value in random_scenarios[sim_index].items():
                    param_value_string = "{:.3g}".format(param_value)
                    param_string += f"{param_name} = {param_value_string}, "
                    base_log_packet[param_name] = param_value
                base_log_packet["Parameters"] = param_string[:-2]

                # Log the goal/safe/unsafe/out of bounds status
                #TODO make sure x_current is coming from turtlebot
                # the other three lines should be fine I think
                x = x_current[sim_index, :].unsqueeze(0)
                is_goal = controller_under_test.dynamics_model.goal_mask(x).all()
                is_safe = controller_under_test.dynamics_model.safe_mask(x).all()
                is_unsafe = controller_under_test.dynamics_model.unsafe_mask(x).all()
                
                #TODO not sure if this needs to be modified but I don't think so
                for measurement_label, value in zip(
                    ["goal", "safe", "unsafe"], [is_goal, is_safe, is_unsafe]
                ):
                    #TODO what's this do?
                    base_log_packet[measurement_label] = value.cpu().numpy().item()

                # Pick out the states to log
                #TODO make sure these state values are being taken from turtlebot
                for i, state_index in enumerate(self.plot_x_indices):
                    state_label = self.plot_x_labels[i]
                    state_value = x_current[sim_index, state_index].cpu().numpy().item()
                    
                    #TODO probably can leave this?
                    log_packet = copy(base_log_packet)
                    log_packet["measurement"] = state_label
                    log_packet["value"] = state_value
                    results_df = results_df.append(log_packet, ignore_index=True)

                # Pick out the controls to log
                #TODO does this need to be modified? I don't think so, but check
                for i, control_index in enumerate(self.plot_u_indices):
                    control_label = self.plot_u_labels[i]
                    u_value = u_current[sim_index, control_index].cpu().numpy().item()
                    
                    #TODO same idea as above, but this is probably fine as is
                    log_packet = copy(base_log_packet)
                    log_packet["measurement"] = control_label
                    log_packet["value"] = u_value
                    results_df = results_df.append(log_packet, ignore_index=True)

                # If this controller supports querying the Lyapunov function, save that
                #TODO this is probably fine as is?
                if hasattr(controller_under_test, "V"):
                    V = controller_under_test.V(x).cpu().numpy().item()  # type: ignore

                    log_packet = copy(base_log_packet)
                    log_packet["measurement"] = "V"
                    log_packet["value"] = V
                    results_df = results_df.append(log_packet, ignore_index=True)

            # Simulate forward using the dynamics
            #TODO make sure to modify for doing only one "simulation"
            for i in range(n_sims):
                xdot = controller_under_test.dynamics_model.closed_loop_dynamics(
                    #TODO modify x_current to be obtained from turtlebot
                    x_current[i, :].unsqueeze(0),
                    #TODO need to send commands to turtlebot
                    u_current[i, :].unsqueeze(0),
                    #TODO get rid of this line? what's this do?
                    random_scenarios[i],
                )
                #TODO get rid of this line and use state given by turtlebot
                x_current[i, :] = x_current[i, :] + delta_t * xdot.squeeze()

        results_df = results_df.set_index("t")
        return results_df
    
    #########################################################################################
    #don't need to worry about plotting for now
#     def plot(
#         self,
#         controller_under_test: "Controller",
#         results_df: pd.DataFrame,
#         display_plots: bool = False,
#     ) -> List[Tuple[str, figure]]:
#         """
#         Plot the results, and return the plot handles. Optionally
#         display the plots.
#         args:
#             controller_under_test: the controller with which to run the experiment
#             display_plots: defaults to False. If True, display the plots (blocks until
#                            the user responds).
#         returns: a list of tuples containing the name of each figure and the figure
#                  object.
#         """
#         # Set the color scheme
#         sns.set_theme(context="talk", style="white")

#         # Plot the state and control trajectories (and V, if it's present)
#         plot_V = "V" in results_df.measurement.values
#         num_plots = len(self.plot_x_indices) + len(self.plot_u_indices)
#         if plot_V:
#             num_plots += 1

#         fig, axs = plt.subplots(num_plots, 1)
#         fig.set_size_inches(10, 4 * num_plots)

#         # Plot all of the states first
#         for i, state_label in enumerate(self.plot_x_labels):
#             ax = axs[i]
#             state_mask = results_df["measurement"] == state_label
#             sns.lineplot(
#                 ax=ax, x="t", y="value", hue="Parameters", data=results_df[state_mask]
#             )
#             ax.set_ylabel(state_label)
#             # Clear the x label since the plots are stacked
#             ax.set_xlabel("")

#         # Then all of the controls
#         for i, control_label in enumerate(self.plot_u_labels):
#             ax = axs[len(self.plot_x_indices) + i]
#             control_mask = results_df["measurement"] == control_label
#             sns.lineplot(
#                 ax=ax, x="t", y="value", hue="Parameters", data=results_df[control_mask]
#             )
#             ax.set_ylabel(control_label)
#             # Clear the x label since the plots are stacked
#             ax.set_xlabel("")

#         # Finally, V (if available)
#         if plot_V:
#             ax = axs[-1]
#             V_mask = results_df["measurement"] == "V"
#             sns.lineplot(
#                 ax=ax, x="t", y="value", hue="Parameters", data=results_df[V_mask]
#             )
#             ax.set_ylabel("$V$")
#             # Clear the x label since the plots are stacked
#             ax.set_xlabel("")

#         # Set one x label for all the stacked plots
#         axs[-1].set_xlabel("t")

#         fig_handle = ("Real Turtlebot (time series)", fig)

#         if display_plots:
#             plt.show()
#             return []
#         else:
#             return [fig_handle]
