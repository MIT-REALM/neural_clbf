import pandas as pd

from neural_clbf.controllers import NeuralCLBFController


def eval_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/turtlebot/commit_dfcbd52/"
    neural_controller = NeuralCLBFController.load_from_checkpoint(
        log_dir + "version_0.ckpt"
    )

    # Modify controller period and simulation time
    neural_controller.controller_period = 0.1
    neural_controller.dynamics_model.dt = 0.01

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_save_to_csv(
        neural_controller, log_dir + "experiments"
    )


def plot_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/turtlebot/commit_dfcbd52/"
    neural_controller = NeuralCLBFController.load_from_checkpoint(
        log_dir + "version_0.ckpt"
    )

    # Set the path to load from
    experiment_dir = log_dir + "experiments/2021-08-20_10_13_56/"

    # Rollout State Space Experiment
    results_df = pd.read_csv(experiment_dir + "/Rollout State Space.csv")
    neural_controller.experiment_suite.experiments[1].plot(
        neural_controller, results_df, display_plots=True
    )


if __name__ == "__main__":
    # eval_turtlebot()
    plot_turtlebot()
