"""A mock experiment for use in testing"""
from typing import cast, List, Tuple, Optional, TYPE_CHECKING
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import torch
import tqdm

from neural_clbf.experiments import Experiment

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller, CLFController  # noqa


class CLFVerificationExperiment(Experiment):
    """An experiment for verifying learned CLFs on a grid.

    WARNING: VERY SLOW. Exponential!!
    """

    def __init__(
        self,
        name: str,
        domain: Optional[List[Tuple[float, float]]] = None,
        n_grid: int = 50,
    ):
        """Initialize an experiment for validating the CLF over a given domain.

        args:
            name: the name of this experiment
            domain: a list of two tuples specifying the plotting range,
                    one for each state dimension.
            n_grid: the number of points in each direction at which to compute V
        """
        super(CLFVerificationExperiment, self).__init__(name)

        # Default to plotting over [-1, 1] in all directions
        if domain is None:
            domain = [(-1.0, 1.0), (-1.0, 1.0)]
        self.domain = domain

        self.n_grid = n_grid

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
        # Sanity check: can only be called on a NeuralCLFController
        if not (
            hasattr(controller_under_test, "V")
            and hasattr(controller_under_test, "solve_CLF_QP")
        ):
            raise ValueError("Controller under test must be a CLFController")

        controller_under_test = cast("CLFController", controller_under_test)

        # Set up a dataframe to store the results
        results_df = pd.DataFrame()

        # Set up the plotting grid
        device = "cpu"
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device  # type: ignore

        # We need a vector for each dimension
        n_dims = controller_under_test.dynamics_model.n_dims
        state_vals = torch.zeros(n_dims, self.n_grid)
        for dim_idx in range(n_dims):
            state_vals[dim_idx, :] = torch.linspace(
                self.domain[dim_idx][0],
                self.domain[dim_idx][1],
                self.n_grid,
                device=device,
            )

        # Loop through the grid, which is a bit tricky since we have to loop through
        # all dimensions, and we don't know how many we have right now. We'll use
        # a cartesian list product to get all points in the grid.
        prog_bar = tqdm.tqdm(product(*state_vals), desc="Validating CLF", leave=True)
        for point in prog_bar:
            x = torch.tensor(point).view(1, -1)

            # Get the value of the CLF
            V = controller_under_test.V(x)

            # Get the goal, safe, or unsafe classification
            is_goal = controller_under_test.dynamics_model.goal_mask(x).all()
            is_safe = controller_under_test.dynamics_model.safe_mask(x).all()
            is_unsafe = controller_under_test.dynamics_model.unsafe_mask(x).all()

            # Get the QP relaxation
            _, r = controller_under_test.solve_CLF_QP(x)
            relaxation = r.max()

            # Store the results
            log_packet = {
                "V": V.cpu().numpy().item(),
                "QP relaxation": relaxation.cpu().numpy().item(),
                "Goal region": is_goal.cpu().numpy().item(),
                "Safe region": is_safe.cpu().numpy().item(),
                "Unsafe region": is_unsafe.cpu().numpy().item(),
            }
            state_list = x.squeeze().cpu().tolist()
            for state_idx, state in enumerate(state_list):
                log_packet[str(state_idx)] = state

            results_df = results_df.append(log_packet, ignore_index=True)

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

        # We want a bar plot showing:
        #   1) Rate of CLF violation (both total and in the invariant set)
        #   2) Fraction of safe states excluded from the invariant set
        #   3) Fraction of unsafe states included in the invariant set

        # Get violation rates
        violation_rate_total = (results_df["QP relaxation"] > 1e-5).mean()
        safe_level = controller_under_test.safe_level  # type:ignore
        invariant = results_df["V"] <= safe_level
        violation_rate_invariant = (
            results_df["QP relaxation"][invariant] > 1e-5
        ).mean()

        # Get the segmentation accuracies
        safe_missed = (results_df[results_df["Safe region"] == 1.0].V > 0.0).mean()
        unsafe_included = (
            results_df[results_df["Unsafe region"] == 1.0].V < 0.0
        ).mean()

        # Plot them
        plotting_df = pd.DataFrame(
            [
                {"Metric": "Valid (total)", "%": 1 - violation_rate_total},
                {"Metric": "Valid (invariant)", "%": 1 - violation_rate_invariant},
                {"Metric": "Safe Set Invariant", "%": 1 - safe_missed},
                {"Metric": "Unsafe Set Invariant", "%": unsafe_included},
            ]
        )
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(12, 8)
        sns.barplot(x="Metric", y="%", data=plotting_df, ax=ax)

        fig_handle = ("CLF Validation", fig)

        if display_plots:
            plt.show()
            return []
        else:
            return [fig_handle]
