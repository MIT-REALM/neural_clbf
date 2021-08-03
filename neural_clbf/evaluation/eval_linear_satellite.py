import pandas as pd

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.experiments import CLFVerificationExperiment


def eval_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/linear_satellite_cbf/commit_318eb38/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Increase the resolution of the grid.
    neural_controller.experiment_suite.experiments[0].n_grid = 500

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_save_to_csv(
        neural_controller, log_dir + "experiments"
    )


def plot_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/linear_satellite_cbf/commit_318eb38/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Load the saved data
    results_df = pd.read_csv(log_dir + "experiments/2021-07-11_17_26_05/V_Contour.csv")

    # Run the experiments and save the results
    neural_controller.experiment_suite.experiments[0].plot(
        neural_controller, results_df, display_plots=True
    )


def verify_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/linear_satellite_cbf/commit_318eb38/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Define an experiment to validate the CBF
    domain = [
        (-2.5, 2.5),
        (-2.5, 2.5),
        (-2.5, 2.5),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    n_grid = 10
    verification_experiment = CLFVerificationExperiment(
        "CLF_Verification", domain, n_grid=n_grid
    )

    # Run the experiments and save the results
    verification_experiment.run_and_plot(neural_controller, display_plots=True)


if __name__ == "__main__":
    eval_linear_satellite()
    # plot_linear_satellite()
    # verify_linear_satellite()
