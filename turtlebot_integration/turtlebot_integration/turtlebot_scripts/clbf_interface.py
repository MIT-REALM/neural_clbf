#!/usr/bin/env python3

import pandas as pd
import numpy as np

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments import CLBFVerificationExperiment
from neural_clbf.experiments import RealTimeSeriesExperiment


def eval_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.

    # loads a checkpoint file contaiing weights runtime etc. Obtained by running the
    # training script
    log_dir = "saved_models/aas/turtlebot/test/"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Increase the resolution of the grid.
    neural_controller.experiment_suite.experiments[0].n_grid = 500

    # take the first experiement from experiment_suite and run it; this
    # is a temporary fix to make the script run a single "experiment"

    #TODO see branch and input proper arguments here
    experiment = RealTimeSeriesExperiment()

    neural_controller.experiment_suite.experiments = [experiment]

    # Run the experiments and save the results
    # neural_controller.experiment_suite.run_all_and_save_to_csv(
    #     neural_controller, log_dir + "experiments"
    # )

    # for now, instead of saving data to a csv, see if we can just
    # get an actual turtlebot to work with the controller.
    #TODO reimplment this (and plotting) once we get the controller
    # talking with the turlebot
    neural_controller.experiment_suite.run_all(
        neural_controller, log_dir + "experiments"
    )

def plot_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/turtlebot/test/"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Load the saved data
    results_df = pd.read_csv(log_dir + "experiments/2021-08-02_20_30_57/V Contour.csv")
    
    # take the first experiement from experiment_suite and run it; this
    # is a temporary fix to make the script run a single "experiment"
    neural_controller.experiment_suite.experiments = [neural_controller.experiment_suite.experiments[0]]

    # Run the experiments and save the results
    neural_controller.experiment_suite.experiments[0].plot(
        neural_controller, results_df, display_plots=True
    )


def verify_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/turtlebot/test/"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Define an experiment to validate the CLBF
    domain = [
        (-2.0, -2.0)
        (-2.0, -2.0)
        (-np.pi/2, np.pi/2)
    ]
    n_grid = 10
    verification_experiment = CLBFVerificationExperiment(
        "CLF_Verification", domain, n_grid=n_grid
    )

    # Run the experiments and save the results
    verification_experiment.run_and_plot(neural_controller, display_plots=True)


if __name__ == "__main__":
    eval_turtlebot()
    plot_turtlebot()