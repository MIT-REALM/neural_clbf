from typing import cast, List, Tuple, TYPE_CHECKING

from matplotlib.pyplot import figure
import pandas as pd
import torch

from neural_clbf.experiments import Experiment

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller, NeuralObsBFController  # noqa


class ObsBFVerificationExperiment(Experiment):
    """An experiment for verifying learned observation BF by sampling"""

    def __init__(
        self,
        name: str,
        n_samples: int = 10000,
    ):
        """Initialize an experiment for validating the BF

        args:
            name: the name of this experiment
            n_samples: the number of points to check
        """
        super(ObsBFVerificationExperiment, self).__init__(name)

        self.n_samples = n_samples

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
        # Sanity check: can only be called on a NeuralObsBFController
        if not (
            hasattr(controller_under_test, "h") and hasattr(controller_under_test, "V")
        ):
            raise ValueError("Controller under test must be a BF controller")

        controller_under_test = cast("NeuralObsBFController", controller_under_test)

        # Set up a dataframe to store the results
        results = []

        # Sample a bunch of points to check from the safe region of state space
        x = controller_under_test.dynamics_model.sample_safe(self.n_samples)
        x = x[controller_under_test.dynamics_model.safe_mask(x)]

        # Evaluate the barrier and lyapunov functions
        obs = controller_under_test.get_observations(x)
        h = controller_under_test.h(x, obs)
        V = controller_under_test.V(x)

        # Get the control input and cost
        controller_under_test.reset_controller(x)
        u, u_cost = controller_under_test.u_(x, obs, h, V)

        results.append(
            {
                "# Samples": x.shape[0],
                "# infeasible": (u_cost > 0).sum().item(),
            }
        )

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
        # Nothing to plot
        pass
