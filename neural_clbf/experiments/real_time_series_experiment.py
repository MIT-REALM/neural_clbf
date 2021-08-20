"""A mock experiment for use in testing"""
from copy import copy
import random
from typing import List, Tuple, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import torch
from torch.nn.functional import linear
import tqdm

from neural_clbf.experiments import Experiment
from neural_clbf.systems.utils import ScenarioList

# turtlebot script imports
from integration.integration.turtlebot_scripts import odometry_status
from integration.integration.turtlebot_scripts.send_command import execute_command
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
            t_sim: float = 150.0,
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

        # Deal with optional parameters
        if self.scenarios is None:
            scenarios = [controller_under_test.dynamics_model.nominal_params]
        else:
            scenarios = self.scenarios

        # Set up a dataframe to store the results
        results_df = pd.DataFrame()

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

        # self.start_x is as shown below in train_turtlebot.py. This will
        # need to be fed in as an argument to this class when this script is called.
        # I commented in for convenience an example starting point:
        # self.start_x = [1.0, 1.0, np.pi/2]
        (self.position, self.rotation) = odometry_status.get_odom(self.listener, self.odom_frame, self.base_frame)


        # temporary placeholder to make the code work, eventually should just replace random
        # scenarios with scenarios since we're not running a bunch of random scenarios, just one.
        random_scenarios = scenarios


        # Make sure everything's on the right device
        device = "cpu"
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device  # type: ignore

        # Simulate!
        delta_t = controller_under_test.dynamics_model.dt
        num_timesteps = int(self.t_sim // delta_t)
        x_current = torch.zeros(1, n_dims).type_as(self.start_x)
        u_current = torch.zeros(1, n_dims).type_as(self.start_x)
        controller_update_freq = int(controller_under_test.controller_period / delta_t)
        prog_bar_range = tqdm.trange(
            0, num_timesteps, desc="Controller Rollout", leave=True
        )

        for tstep in prog_bar_range:
            
            (self.position, self.rotation) = odometry_status.get_odom(self.listener, self.odom_frame, self.base_frame)
            x_current[0, :] = torch.tensor([self.position.x, self.position.y, self.rotation]) + self.start_x

            # Get the control input at the current state if it's time
            if tstep % controller_update_freq == 0:

                # TODO when Charles sends the updated code, make sure to change this
                # back to controller_under_test.u(x_current) to make sure it works
                u_current = controller_under_test.dynamics_model.u_nominal(x_current)
                # set the output command to the command obtained from the
                # dynamics model
                linear_command = u_current[0][0].item()
                print(linear_command)
                angular_command = u_current[0][1].item()

                # call the function that sends the commands to the turtlebot

                # pull the control limits from the turtlebot system file
                (upper_command_limit, _)= controller_under_test.dynamics_model.control_limits
                execute_command(self.command_publisher, self.move_command, linear_command, angular_command, self.position, self.rotation, upper_command_limit)

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

                
                print(x_current)
                print(u_current)
                print(is_goal)
                print(is_safe)
            



        results_df = results_df.set_index("t")
        return results_df

    #########################################################################################
    # don't need to worry about plotting for now
    def plot():
        pass




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