"""An experiment suite manages a collection of experiments, allowing the user to
run each experiment.
"""
from datetime import datetime
import os
from typing import List, Optional, Tuple, TYPE_CHECKING

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning.loggers import LightningLoggerBase

from .experiment import Experiment

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller  # noqa


class ExperimentSuite(object):
    """docstring for ExperimentSuite"""

    def __init__(self, experiments: List[Experiment]):
        """
        Create an ExperimentSuite: an object for managing a collection of experiments.

        args:
            experiments: a list of Experiment objects comprising the suite
        """
        super(ExperimentSuite, self).__init__()
        self.experiments = experiments

    def run_all(self, controller_under_test: "Controller") -> List[pd.DataFrame]:
        """Run all experiments in the suite and return the data from each

        args:
            controller_under_test: the controller with which to run the experiments
        returns:
            a list of DataFrames, one for each experiment
        """
        results = []
        for experiment in self.experiments:
            results.append(experiment.run(controller_under_test))

        return results

    def run_all_and_save_to_csv(
        self, controller_under_test: "Controller", save_dir: str
    ):
        """Run all experiments in the suite and save the results in one directory.

        Results will be saved in a subdirectory save_dir/{timestamp}/...

        args:
            controller_under_test: the controller with which to run the experiments
            save_dir: the path to the directory in which to save the results
        returns:
            a list of DataFrames, one for each experiment
        """
        # Make sure the given directory exists; create it if it does not
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get the subdirectory name
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y-%m-%d_%H_%M_%S")
        subdirectory_path = f"{save_dir}/{timestamp}"

        # Run and save all experiments (these will create subdirectory if it does not
        # already exist)
        for experiment in self.experiments:
            experiment.run_and_save_to_csv(controller_under_test, subdirectory_path)

    def run_all_and_plot(
        self, controller_under_test: "Controller", display_plots: bool = False
    ) -> List[Tuple[str, figure]]:
        """
        Run all experiments, plot the results, and return the plot handles. Optionally
        display the plots.

        args:
            controller_under_test: the controller with which to run the experiments
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """
        # Create a place to store all the returned handles
        fig_handles = []

        # Run each experiment, plot, and store the handles
        for experiment in self.experiments:
            fig_handles += experiment.run_and_plot(controller_under_test, display_plots)

        return fig_handles

    def run_all_and_log_plots(
        self,
        controller_under_test: "Controller",
        logger: LightningLoggerBase,
        log_epoch: int,
        plot_tag: Optional[str] = None,
    ):
        """Run all experiments, plot the results, and log the plots using the provided
        logger

        args:
            controller_under_test: the controller with which to run the experiments
            logger: the logger to use for saving the plots
            log_epoch: the current log epoch
            plot_tag: if provided, format plot names as "plot_name::plot_tag"
        """
        # Handle default argument
        if plot_tag is None:
            plot_tag = ""
        else:
            plot_tag = "::" + plot_tag

        # Run the experiments and get the plot handles
        fig_handles = self.run_all_and_plot(controller_under_test, display_plots=False)

        # Log each plot
        for plot_name, figure_handle in fig_handles:
            logger.experiment.add_figure(
                plot_name + plot_tag, figure_handle, global_step=log_epoch
            )
        logger.experiment.close()
        logger.experiment.flush()

        # Clean up by closing each plot
        for _, figure_handle in fig_handles:
            plt.close(figure_handle)
