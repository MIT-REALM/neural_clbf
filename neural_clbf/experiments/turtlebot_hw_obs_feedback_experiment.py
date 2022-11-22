"""An experiment on TurtleBot HW with a laser scanner"""
from typing import List, Tuple, TYPE_CHECKING

import rospy

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import pandas as pd
import torch

from neural_clbf.experiments import Experiment

# turtlebot script imports
from integration.integration.turtlebot_scripts import odometry_status
from integration.integration.turtlebot_scripts.send_command import execute_command
from integration.integration.turtlebot_scripts.laser_data import LidarMonitor
import os


if TYPE_CHECKING:
    from neural_clbf.controllers import Controller  # noqa
    from neural_clbf.controllers import NeuralObsBFController  # noqa


class TurtlebotHWObsFeedbackExperiment(Experiment):
    """An experiment for running observation-feedback controllers on the turtlebot.

    Plots trajectories as a function of time.
    """

    def __init__(
        self,
        name: str,
        command_publisher,
        rate,
        listener,
        move_command,
        odom_frame,
        base_frame,
        lidar_monitor: LidarMonitor,
        start_x: torch.Tensor,
        t_sim: float = 10.0,
    ):
        """
        Initialize an experiment for controller performance on turtlebot.

        args:

        """
        super(TurtlebotHWObsFeedbackExperiment, self).__init__(name)

        # clbf parameters
        self.start_x = start_x
        self.t_sim = t_sim

        # turtlebot interface parameters
        self.rate = rate
        self.command_publisher = command_publisher
        self.listener = listener
        self.move_command = move_command
        self.odom_frame = odom_frame
        self.base_frame = base_frame
        self.lidar_monitor = lidar_monitor

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
        os.system("timeout 3 rostopic pub /reset std_msgs/Empty '{}'")

        # Set up a dataframe to store the results
        results = []

        # Save these for convenience
        n_dims = controller_under_test.dynamics_model.n_dims
        assert n_dims == 3, "Dynamics incompatible with TurtleBot"

        # get intial state from odometry
        (self.position, self.rotation) = odometry_status.get_odom(
            self.listener, self.odom_frame, self.base_frame
        )

        # Execute!
        delta_t = controller_under_test.controller_period
        x_current = torch.zeros(1, n_dims).type_as(self.start_x)
        u_current = torch.ones(1, n_dims).type_as(self.start_x)
        current_time = 0.0
        # r = rospy.Rate(1/delta_t)
        r = rospy.Rate(20)

        # Reset the controller if necessary
        if hasattr(controller_under_test, "reset_controller"):
            controller_under_test.reset_controller(x_current)  # type: ignore

        while not rospy.is_shutdown():
            # Get the position from odometry and add the offset
            (self.position, self.rotation) = odometry_status.get_odom(
                self.listener, self.odom_frame, self.base_frame
            )
            x_current[0, :] = (
                torch.tensor([self.position.x, self.position.y, self.rotation])
                + self.start_x
            )

            # Get the control input at the current state
            obs_current = self.lidar_monitor.last_scan.clone().detach()
            u_current = controller_under_test.u_from_obs(  # type: ignore
                x_current, obs_current
            )

            # set the output command to the command obtained from the
            # dynamics model
            linear_command = u_current[0][0].item()
            angular_command = u_current[0][1].item()

            # pull the control limits from the turtlebot system file
            u_max, _ = controller_under_test.dynamics_model.control_limits

            # call the function that sends the commands to the turtlebot
            execute_command(
                self.command_publisher,
                self.move_command,
                linear_command,
                angular_command,
                self.position,
                self.rotation,
                u_max,
            )

            # Log the current state and control
            log_packet = {"t": current_time}
            log_packet["$x$"] = self.position.x
            log_packet["$y$"] = self.position.y
            log_packet["$\\theta$"] = self.rotation
            log_packet["$v$"] = linear_command
            log_packet["$\\omega$"] = angular_command

            # If this controller supports querying the Lyapunov function, save that
            if hasattr(controller_under_test, "V"):
                V = (
                    controller_under_test.V(x_current)  # type: ignore
                    .cpu()
                    .numpy()
                    .item()
                )
                log_packet["V"] = V
            # If this controller supports querying the barrier function, save that
            if hasattr(controller_under_test, "h"):
                h = (
                    controller_under_test.h(x_current, obs_current)  # type: ignore
                    .cpu()
                    .numpy()
                    .item()
                )
                log_packet["h"] = h

            results.append(log_packet)

            r.sleep()
            current_time += delta_t

        results_df = pd.DataFrame(results)
        results_df = results_df.set_index("t")
        return results_df

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
            results_df: a DataFrame of results, as returned by `self.run`
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """
        # Set the color scheme
        sns.set_theme(context="talk", style="white")

        # Plot the state trajectories
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)
        sns.lineplot(
            ax=ax,
            x="$x$",
            y="$y$",
            style="Parameters",
            hue="Simulation",
            data=results_df,
        )

        # Plot the environment
        controller_under_test.dynamics_model.plot_environment(ax)

        fig_handle = ("HW Rollout (state space)", fig)

        if display_plots:
            plt.show()
            return []
        else:
            return [fig_handle]
