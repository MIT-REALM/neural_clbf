import numpy as np

import torch

from neural_clbf.controllers import NeuralObsBFController
import neural_clbf.evaluation.scenes as scene_utils


@torch.no_grad()
def eval_and_plot_turtlebot_room():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/perception/turtlebot2d/commit_26f34ff/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Get the experiments
    contour_experiment = neural_controller.experiment_suite.experiments[0]
    rollout_experiment = neural_controller.experiment_suite.experiments[1]

    # Modify contour parameters
    contour_experiment.n_grid = 30
    contour_experiment.domain = [(-4.0, 0.0), (-2.0, 2.0)]

    # Modify rollout parameters
    rollout_experiment.t_sim = 10
    rollout_experiment.start_x = torch.tensor(
        [
            # Start from same room as goal (OK, 10s)
            [-2.5, -2.5, np.pi / 2],
            [-4.0, 0.0, 0.0],
            # # Start from table room (OK, 20s)
            # [-13.5, 1.0, 0.0],
            # [-11.83, -4.8, 0.0],
            # # Start from chair room (OK, 80)
            # [-13.5, -13.5, 0.0],
            # [-7.0, -8.0, 0.0],
            # Start from chair room (testing)
            # [-1.0, -13.5, 0.0],  # (OK, 80)
            # [-3.0, -12, 0.0],  # (OK, 200)
            # [-3.8, -11, 0.0],  # (OK, 100)
        ]
    )
    neural_controller.lookahead_grid_n = 8
    neural_controller.controller_period = 0.1
    neural_controller.dynamics_model.dt = 0.01
    neural_controller.lookahead_dual_penalty = 1e3
    # neural_controller.debug_mode_exploratory = True
    # neural_controller.debug_mode_goal_seeking = True

    # Modify scene
    scene = scene_utils.room_4()
    neural_controller.dynamics_model.scene = scene

    # Run the experiments and plot
    rollout_experiment.run_and_plot(neural_controller, display_plots=True)
    # contour_experiment.run_and_plot(neural_controller, display_plots=True)


def eval_and_plot_turtlebot_bugtrap():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/perception/turtlebot2d/commit_04c9147/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Get the experiments
    cbf_contour_experiment = neural_controller.experiment_suite.experiments[0]
    clf_contour_experiment = neural_controller.experiment_suite.experiments[1]
    rollout_experiment = neural_controller.experiment_suite.experiments[2]

    # Modify contour parameters
    cbf_contour_experiment.n_grid = 30
    cbf_contour_experiment.domain = [(-4.0, 0.0), (-2.0, 2.0)]
    clf_contour_experiment.n_grid = 30
    clf_contour_experiment.domain = [(-4.0, 0.0), (-2.0, 2.0)]

    # Modify rollout parameters
    rollout_experiment.t_sim = 4
    rollout_experiment.start_x = torch.tensor(
        [
            [-3.0, -0.1, 0.0],
        ]
    )
    neural_controller.lookahead_grid_n = 8
    neural_controller.controller_period = 0.1
    neural_controller.dynamics_model.dt = 0.01
    neural_controller.lookahead_dual_penalty = 1e3
    # neural_controller.debug_mode_exploratory = True
    # neural_controller.debug_mode_goal_seeking = True

    # Modify scene
    scene = scene_utils.bugtrap()
    neural_controller.dynamics_model.scene = scene

    # Run the experiments and plot
    rollout_experiment.run_and_plot(neural_controller, display_plots=True)
    # cbf_contour_experiment.run_and_plot(neural_controller, display_plots=True)


def eval_and_plot_turtlebot_training():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/perception/turtlebot2d/commit_04c9147/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Get the experiment
    rollout_experiment = neural_controller.experiment_suite.experiments[2]

    # Modify rollout parameters
    rollout_experiment.t_sim = 4
    neural_controller.lookahead_grid_n = 8
    neural_controller.controller_period = 0.1
    neural_controller.dynamics_model.dt = 0.01
    neural_controller.lookahead_dual_penalty = 1e3
    # neural_controller.debug_mode_exploratory = True
    # neural_controller.debug_mode_goal_seeking = True

    # Run the experiments and plot
    rollout_experiment.run_and_plot(neural_controller, display_plots=True)
    # cbf_contour_experiment.run_and_plot(neural_controller, display_plots=True)


if __name__ == "__main__":
    # eval_and_plot_turtlebot_room()
    # eval_and_plot_turtlebot_bugtrap()
    eval_and_plot_turtlebot_training()
