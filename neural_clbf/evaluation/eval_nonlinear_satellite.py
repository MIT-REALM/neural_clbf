import pandas as pd

from neural_clbf.controllers import NeuralCBFController


def eval_nonlinear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/nonlinear_satellite/commit_bb05c2c/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Increase the resolution of the grid
    neural_controller.experiment_suite.experiments[0].n_grid = 500

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_save_to_csv(
        neural_controller, log_dir + "experiments"
    )


def plot_nonlinear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/nonlinear_satellite/commit_bb05c2c/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Increase the resolution of the grid
    neural_controller.experiment_suite.experiments[0].n_grid = 500

    # Load the saved data
    data_path = "saved_models/aas/nonlinear_satellite/commit_bb05c2c/experiments/"
    data_path += "2021-07-07_10:47:30/V Contour.csv"
    results_df = pd.read_csv(data_path)

    # Plot
    neural_controller.experiment_suite.experiments[0].plot(
        neural_controller, results_df, display_plots=True
    )


if __name__ == "__main__":
    plot_nonlinear_satellite()
