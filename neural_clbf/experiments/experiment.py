"""Define a generic experiment that can be extended to other control problems.

An "experiment" is anything that tests the behavior of a controller. Experiments should
be limited to testing one thing about the controller; for example, simulating a rollout
or plotting the Lyapunov function on a grid.

Each experiment should be able to do a number of things:
    1) Run the experiment on a given controller
    2) Save the results of that experiment to a CSV, respecting the tidy data principle
    3) Plot the results of that experiment and return the plot handle, with the option
       to display the plot.
"""
from abc import (
    ABC,
    abstractmethod,
)
import os
from typing import List, Tuple, TYPE_CHECKING

from matplotlib.pyplot import figure
import pandas as pd

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller


class Experiment(ABC):
    """A generic control experiment"""

    def __init__(self, name: str):
        """Initialize a generic experiment for a controller

        args:
            name: the name for this experiment
        """
        super(Experiment, self).__init__()
        self.name = name

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def run_and_plot(
        self, controller_under_test: "Controller", display_plots: bool = False
    ) -> List[Tuple[str, figure]]:
        """
        Run the experiment, plot the results, and return the plot handles. Optionally
        display the plots.

        args:
            controller_under_test: the controller with which to run the experiment
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """
        results_df = self.run(controller_under_test)
        return self.plot(controller_under_test, results_df, display_plots)

    def run_and_save_to_csv(self, controller_under_test: "Controller", save_dir: str):
        """
        Run the experiment and save the results to a file.

        Results will be saved in savedir/{self.name}.csv

        args:
            controller_under_test: the controller with which to run the experiment
            save_dir: the path to the directory in which to save the results.
        """
        # Make sure the given directory exists; create it if it does not
        os.makedirs(save_dir, exist_ok=True)

        # Get the filename from the experiment name
        filename = f"{save_dir}/{self.name}.csv"

        # Get the results and save
        results = self.run(controller_under_test)
        results.to_csv(filename, index=False)
