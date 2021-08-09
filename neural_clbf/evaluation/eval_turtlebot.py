import pandas as pd

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.experiments import CLFVerificationExperiment
import numpy as np


def eval_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/turtlebot_CLBF/commit_318eb38/"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Increase the resolution of the grid.
    neural_controller.experiment_suite.experiments[0].n_grid = 500

    # TODO fix this later, but duct tape solution for now. This turns the list of
    # experiments into a single element list with the first experiment; we only need
    # one "experiment" for a real life test.
    neural_controller.experiment_suite.experiments = [neural_controller.experiment_suite.experiments[0]]

    # Run the experiment and save the results
    neural_controller.experiment_suite.run_all_and_save_to_csv(
        neural_controller, log_dir + "experiments"
    )


def plot_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/turtlebot_CLBF/commit_318eb38/"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Load the saved data
    results_df = pd.read_csv(log_dir + "experiments/2021-07-11_17_26_05/V_Contour.csv")

    # Run the experiments and save the results
    neural_controller.experiment_suite.experiments[0].plot(
        neural_controller, results_df, display_plots=True
    )


def verify_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/turtlebot_CLBF/commit_318eb38/"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Define an experiment to validate the CLBF
    domain = [
        (-2.0, -2.0) #x
        (-2.0, -2.0) #y
        (-np.pi/2, np.pi/2) #theta
    ]
    n_grid = 10
    verification_experiment = CLFVerificationExperiment(
        "CLBF_Verification", domain, n_grid=n_grid
    )

    # Run the experiments and save the results
    verification_experiment.run_and_plot(neural_controller, display_plots=True)


if __name__ == "__main__":
    eval_turtlebot()
    plot_turtlebot()
    verify_turtlebot()
