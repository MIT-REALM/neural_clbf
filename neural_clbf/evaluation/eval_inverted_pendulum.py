import torch
import matplotlib
from neural_clbf.controllers import NeuralCLBFController


matplotlib.use('TkAgg')


def plot_inverted_pendulum():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "saved_models/review/inverted_pendulum_clf.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Update parameters
    neural_controller.experiment_suite.experiments[1].start_x = torch.tensor(
        [
            [1.5, 1.5],
            [0.9, 1.5],
            [0.3, 1.5],
            [0.0, 1.5],
            [-0.3, 1.5],
            [-0.9, 1.5],
            [-1.5, 1.5],
            [1.5, -1.5],
            [0.9, -1.5],
            [0.3, -1.5],
            [0.0, -1.5],
            [-0.3, -1.5],
            [-0.9, -1.5],
            [-1.5, -1.5],
        ]
    )

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )


if __name__ == "__main__":
    # eval_inverted_pendulum()
    plot_inverted_pendulum()
