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

# turtlebot script imports
import turtlebot_scripts.odometry_status as odometry_status
from neural_clbf.turtlebot_scripts.send_command import execute_command
import os

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller  # noqa


class RealTimeSeriesExperiment(Experiment):
    """An experiment for plotting actual
    performance of controller on turtlebot.

    Plots trajectories as a function of time.
    """

    def __init__(
            self,

            # turtlebot interface parameters
            command_publisher,
            rate,
            listener,
            move_command,
            odom_frame,
            base_frame,

            # clbf parameters
            name: str,
            start_x: torch.Tensor,
            plot_x_indices: List[int],
            plot_x_labels: List[str],
            plot_u_indices: List[int],
            plot_u_labels: List[str],
            scenarios: Optional[ScenarioList] = None,
            # n_sims_per_start: int = 5,
            # placeholder runtime, will likely need to either restructure to avoid a set amount of time
            # or change this value in some way
            t_sim: float = 30.0,
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

        # clbf parameters
        self.start_x = start_x
        self.plot_x_indices = plot_x_indices
        self.plot_x_labels = plot_x_labels
        self.plot_u_indices = plot_u_indices
        self.plot_u_labels = plot_u_labels
        self.scenarios = scenarios
        # self.n_sims_per_start = n_sims_per_start
        self.t_sim = t_sim

        # turtlebot interface parameters
        self.rate = rate
        self.command_publisher = command_publisher
        self.listener = listener
        self.move_command = move_command
        self.odom_frame = odom_frame
        self.base_frame = base_frame

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
        # reset turtlebot odometry
        # TODO can get rid of this once we get a positioning system
        os.system("timeout 3 rostopic pub /reset std_msgs/Empty '{}'")

        # Each scenario has a specific set of parameters. In this case,
        # it's the radius of the turtlebot's wheels and the distance between them
        # (R and L, respectively).

        # note that we don't need to add a scenario in here because the line below
        # will use the scenario taken from the train_turtlebot.py file if no specific
        # scenario to run is given

        if self.scenarios is None:
            scenarios = [controller_under_test.dynamics_model.nominal_params]
        else:
            scenarios = self.scenarios

        # Set up a dataframe to store the results
        results_df = pd.DataFrame()

        # Compute the number of simulations to run
        # n_sims = self.n_sims_per_start * self.start_x.shape[0]

        # since we're only running one simulation for real life, used
        # a hardcoded value for now
        #TODO see what exactly n_sims_per_start does vs. n_sims
        n_sims = 1
        self.n_sims_per_start = 1

        # Determine the parameter range to sample from
        # this pulls the min and max values from the turtlebot model script
        # so no need to modify this
        parameter_ranges = {}
        for param_name in scenarios[0].keys():
            param_max = max([s[param_name] for s in scenarios])
            param_min = min([s[param_name] for s in scenarios])
            parameter_ranges[param_name] = (param_min, param_max)

        # Generate a tensor of start states
        n_dims = controller_under_test.dynamics_model.n_dims
        n_controls = controller_under_test.dynamics_model.n_controls
        # ARGUMENT
        # self.start_x is as shown below in train_turtlebot.py. This will
        # need to be fed in as an argument to this class when this script is called.
        # commented in for convenience the starting point:
        # self.start_x = [1.0, 1.0, np.pi/2]
        x_sim_start = torch.zeros(n_sims, n_dims).type_as(self.start_x)
        (self.position, self.rotation) = odometry_status.get_odom(self.listener, self.odom_frame, self.base_frame)

        # the controller tries to move to the origin. Thus, we need to 
        # modify the odometry values a bit to make the controller 
        # think it doesn't start at the origin.
        # In this case, we make the starting position roughly be (-1,-1)
        # meaning the turtlebot should move to (1,1) to get to the "origin"
        x_sim_start[0] = [self.position.x, self.position.y, self.rotation]
        x_sim_start += self.start_x

        # commented out; don't think we need it
        # for i in range(0, self.start_x.shape[0]):
        #     for j in range(0, self.n_sims_per_start):
        #         x_sim_start[i * self.n_sims_per_start + j, :] = self.start_x[i, :]

        # Generate a random scenario for each rollout from the given scenarios
        # commented out because we don't want random scenarios, just a single predefined
        # scenario.

        # temporary placeholder to make the code work, eventually should just replace random
        # scenarios with scenarios since we're not running a bunch of random scenarios, just one.
        random_scenarios = scenarios

        # random_scenarios = []
        # for i in range(n_sims):
        #     random_scenario = {}
        #     for param_name in scenarios[0].keys():
        #         param_min = parameter_ranges[param_name][0]
        #         param_max = parameter_ranges[param_name][1]
        #         random_scenario[param_name] = random.uniform(param_min, param_max)
        #     random_scenarios.append(random_scenario)

        # Make sure everything's on the right device
        device = "cpu"
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device  # type: ignore

        # Simulate!
        delta_t = controller_under_test.dynamics_model.dt

        num_timesteps = int(self.t_sim // delta_t)

        # should be fine as is; this just sets the initial position to
        # whatever starting position we use
        x_current = x_sim_start.to(device)

        # initial command matrix of zeros; no need to change this line
        u_current = torch.zeros(x_sim_start.shape[0], n_controls, device=device)

        controller_update_freq = int(controller_under_test.controller_period / delta_t)

        prog_bar_range = tqdm.trange(
            0, num_timesteps, desc="Controller Rollout", leave=True
        )

        # for now, we'll keep the set "simulation" time, but this might need to be changed if it
        # doesn't work or is just not optimal

        for tstep in prog_bar_range:
            # Get the control input at the current state if it's time
            if tstep % controller_update_freq == 0:
                u_current = controller_under_test.u(x_current)

            # Log the current state and control
            base_log_packet = {"t": tstep * delta_t}

            # Include the parameters
            param_string = ""
            for param_name, param_value in random_scenarios[0].items():
                param_value_string = "{:.3g}".format(param_value)
                param_string += f"{param_name} = {param_value_string}, "
                base_log_packet[param_name] = param_value
            base_log_packet["Parameters"] = param_string[:-2]

            # Log the goal/safe/unsafe/out of bounds status
            x = x_current[0, :].unsqueeze(0)
            is_goal = controller_under_test.dynamics_model.goal_mask(x).all()
            is_safe = controller_under_test.dynamics_model.safe_mask(x).all()
            is_unsafe = controller_under_test.dynamics_model.unsafe_mask(x).all()

            for measurement_label, value in zip(
                    ["goal", "safe", "unsafe"], [is_goal, is_safe, is_unsafe]
            ):
                base_log_packet[measurement_label] = value.cpu().numpy().item()

            # Pick out the states to log
            for i, state_index in enumerate(self.plot_x_indices):
                state_label = self.plot_x_labels[i]
                state_value = x_current[0, state_index].cpu().numpy().item()

                # TODO probably can leave this?
                log_packet = copy(base_log_packet)
                log_packet["measurement"] = state_label
                log_packet["value"] = state_value
                results_df = results_df.append(log_packet, ignore_index=True)

            # Pick out the controls to log
            for i, control_index in enumerate(self.plot_u_indices):
                control_label = self.plot_u_labels[i]
                u_value = u_current[0, control_index].cpu().numpy().item()

                log_packet = copy(base_log_packet)
                log_packet["measurement"] = control_label
                log_packet["value"] = u_value
                results_df = results_df.append(log_packet, ignore_index=True)

            # If this controller supports querying the Lyapunov function, save that
            if hasattr(controller_under_test, "V"):
                V = controller_under_test.V(x).cpu().numpy().item()  # type: ignore

                log_packet = copy(base_log_packet)
                log_packet["measurement"] = "V"
                log_packet["value"] = V
                results_df = results_df.append(log_packet, ignore_index=True)

            # Move forward using the dynamics
            # this block seems good to go
            xdot = controller_under_test.dynamics_model.closed_loop_dynamics(
                x_current[0, :].unsqueeze(0),
                u_current[0, :].unsqueeze(0),

                random_scenarios[0],
            )

            # get x, y position and current rotation from odometry
            # TODO this will need to be fixed when we use a positioning system
            (self.position, self.rotation) = odometry_status.get_odom(self.listener, self.odom_frame, self.base_frame)

            # update the current position
            # since the odometry starts off with (0,0,0), we need to add to
            # the odometric measurements the actual starting position
            x_current[0, :] = [self.position.x, self.position.y, self.rotation] + self.start_x

            # set the output command to the command obtained from the
            # dynamics model
            linear_command = xdot[0]
            angular_command = xdot[1]

            # call the function that sends the commands to the turtlebot
            execute_command(self.command_publisher, self.move_command, linear_command, angular_command, self.position, self.rotation)

        results_df = results_df.set_index("t")
        return results_df

    #########################################################################################
    # don't need to worry about plotting for now
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