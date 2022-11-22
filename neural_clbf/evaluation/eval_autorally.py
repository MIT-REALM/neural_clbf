import matplotlib
from neural_clbf.controllers import NeuralCLBFController


matplotlib.use('TkAgg')


def plot_autorally():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "saved_models/autorally/v0.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Tweak controller params
    neural_controller.clf_relaxation_penalty = 1e3
    neural_controller.controller_period = 0.05

    # Tweak experiment params
    neural_controller.experiment_suite.experiments[1].t_sim = 10.0

    # Run the experiments and save the results
    neural_controller.experiment_suite.experiments[1].run_and_plot(
        neural_controller, display_plots=True
    )


if __name__ == "__main__":
    plot_autorally()
