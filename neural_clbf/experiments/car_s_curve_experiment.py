"""A mock experiment for use in testing"""
from copy import copy
from typing import List, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm

from neural_clbf.experiments import Experiment
from neural_clbf.systems import STCar, KSCar

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller  # noqa


class CarSCurveExperiment(Experiment):
    """An experiment for plotting tracking performance of car controllers"""

    def __init__(
        self,
        name: str,
        t_sim: float = 5.0,
    ):
        """Initialize an experiment for testing car tracking performance.

        args:
            name: the name of this experiment
            t_sim: the amount of time to simulate for
        """
        super(CarSCurveExperiment, self).__init__(name)

        # Save parameters
        self.t_sim = t_sim

    @torch.no_grad()
    def run(self, controller_under_test: "Controller") -> pd.DataFrame:
        """
        Run the experiment.

        args:
            controller_under_test: the controller with which to run the experiment.
                                   For this experiment, must be affiliated with a
                                   car dynamics model
        returns:
            a pandas DataFrame containing the results of the experiment, in tidy data
            format (i.e. each row should correspond to a single observation from the
            experiment).
        """
        # Make sure that the controller under test has a car dynamics model
        assert isinstance(controller_under_test.dynamics_model, KSCar) or isinstance(
            controller_under_test.dynamics_model, STCar
        ), "Controller must have a KSCar or STCar dynamics model"

        # Set up a dataframe to store the results
        results_df = pd.DataFrame()

        device = "cpu"
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device  # type: ignore

        # Simulate!
        delta_t = controller_under_test.dynamics_model.dt
        T = int(self.t_sim // delta_t)

        n_dims = controller_under_test.dynamics_model.n_dims
        n_controls = controller_under_test.dynamics_model.n_controls
        x_current = torch.zeros(1, n_dims, device=device)
        u_current = torch.zeros(1, n_controls).type_as(x_current)

        # And create a place to store the reference path
        params = copy(controller_under_test.dynamics_model.nominal_params)
        params["omega_ref"] = 0.3
        x_ref = 0.0
        y_ref = 0.0
        psi_ref = 1.0
        omega_ref = 0.0

        controller_update_freq = int(controller_under_test.controller_period / delta_t)
        prog_bar_range = tqdm.trange(0, T, desc="S-Curve", leave=True)
        for tstep in prog_bar_range:
            # Update the reference path to trace an S curve
            omega_ref = 1.5 * np.sin(tstep * delta_t)
            psi_ref += delta_t * omega_ref
            pt = copy(params)
            pt["omega_ref"] = omega_ref
            pt["psi_ref"] = psi_ref
            x_ref += delta_t * pt["v_ref"] * np.cos(psi_ref)
            y_ref += delta_t * pt["v_ref"] * np.sin(psi_ref)

            # Update the controller if it's time
            if tstep % controller_update_freq == 0:
                u_current = controller_under_test.u(x_current)

            # Update the dynamics
            xdot = controller_under_test.dynamics_model.closed_loop_dynamics(
                x_current,
                u_current,
                pt,
            )
            x_current += delta_t * xdot.squeeze()

            # Log the position at this state to the dataframe
            base_log_packet = {"t": tstep * delta_t}
            measurement_labels = ["$x_{ref}$", "$y_{ref}$", "$x$", "$y$"]
            x_err = x_current[0, controller_under_test.dynamics_model.SXE]
            y_err = x_current[0, controller_under_test.dynamics_model.SYE]
            x = x_ref + x_err * np.cos(psi_ref) - y_err * np.sin(psi_ref)
            y = y_ref + x_err * np.sin(psi_ref) + y_err * np.cos(psi_ref)
            measurements = [
                x_ref,
                y_ref,
                x.cpu().numpy().item(),
                y.cpu().numpy().item(),
            ]
            for label, value in zip(measurement_labels, measurements):
                log_packet = copy(base_log_packet)
                log_packet["measurement"] = label
                log_packet["value"] = value
                results_df = results_df.append(log_packet, ignore_index=True)

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
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """

        # Set the color scheme
        sns.set_theme(context="talk", style="white")

        # Plot the reference and tracking trajectories
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)

        tracking_trajectory_color = sns.color_palette("pastel")[1]
        x_ref = results_df[results_df.measurement == "$x_{ref}$"]
        y_ref = results_df[results_df.measurement == "$y_{ref}$"]
        x = results_df[results_df.measurement == "$x$"]
        y = results_df[results_df.measurement == "$y$"]
        # import pdb; pdb.set_trace()
        ax.plot(
            x_ref.value,
            y_ref.value,
            linestyle="dotted",
            label="Reference",
            color="black",
        )
        ax.plot(
            x.value,
            y.value,
            linestyle="solid",
            label="Controller",
            color=tracking_trajectory_color,
        )
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.legend()
        ax.set_aspect("equal")

        fig_handle = ("S-Curve Tracking", fig)

        if display_plots:
            plt.show()
            return []
        else:
            return [fig_handle]
