"""A mock experiment for use in testing"""
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd

from neural_clbf.experiments import Experiment
from neural_clbf.controllers import Controller


class MockExperiment(Experiment):
    """A mock experiment for use during testing"""

    def run(self, controller_under_test: Controller) -> pd.DataFrame:
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
        # Return an empty dataframe
        results_df = pd.DataFrame({"t": [0, 1, 2, 3], "x": [0, 1, 2, 3]})
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
        # Make a plot
        fig, axes = plt.subplots(1, 1)

        axes.plot(results_df["t"], results_df["x"])

        fig_handle = ("Test Plot", fig)

        if display_plots:
            plt.show()
            return []
        else:
            return [fig_handle]
