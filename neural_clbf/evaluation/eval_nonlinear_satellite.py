import pandas as pd
import torch

from neural_clbf.controllers import NeuralCBFController


def load_and_tweak_cbf():
    """Load the CBF from a checkpoint file and tweak its output bias manually"""
    log_dir = "saved_models/aas/nonlinear_satellite/commit_ba1914d/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v0.ckpt")
    neural_controller.V_nn.output_linear.bias.sub_(0.2)

    return neural_controller, log_dir


@torch.no_grad()
def eval_nonlinear_satellite():
    neural_controller, log_dir = load_and_tweak_cbf()

    # Increase the resolution of the grid
    neural_controller.experiment_suite.experiments[0].n_grid = 500

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_save_to_csv(
        neural_controller, log_dir + "experiments"
    )


@torch.no_grad()
def plot_nonlinear_satellite():
    neural_controller, log_dir = load_and_tweak_cbf()

    # Increase the resolution of the grid
    neural_controller.experiment_suite.experiments[0].n_grid = 500

    # Load the saved data
    data_path = "saved_models/aas/nonlinear_satellite/commit_ba1914d/experiments/"
    data_path += "2021-07-07_17:03:55/V Contour.csv"
    results_df = pd.read_csv(data_path)

    # Plot
    neural_controller.experiment_suite.experiments[0].plot(
        neural_controller, results_df, display_plots=True
    )


if __name__ == "__main__":
    plot_nonlinear_satellite()
