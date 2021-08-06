import numpy as np

from shapely.geometry import box
import torch

from neural_clbf.controllers import NeuralObsBFController

from neural_clbf.systems.planar_lidar_system import Scene


@torch.no_grad()
def eval_and_plot_linear_satellite():
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
    rollout_experiment.t_sim = 5
    rollout_experiment.start_x = torch.tensor(
        [
            [-2.5, -2.5, np.pi / 2],
            [-4.0, 0.0, 0.0],
            # [-2.0, 0.0, 0.0],
        ]
    )
    neural_controller.lookahead_grid_n = 8
    neural_controller.controller_period = 0.1
    neural_controller.dynamics_model.dt = 0.01
    neural_controller.lookahead_dual_penalty = 1e3
    # neural_controller.debug_mode_exploratory = True
    # neural_controller.debug_mode_goal_seeking = True

    # Modify scene
    scene = Scene([])
    scene.add_walls(10.0)
    scene.add_obstacle(box(-1.1, -1.0, -0.9, 1.0))
    # scene.add_obstacle(rotate(box(-1.1, -0.6, -0.9, 0.6), 1, use_radians=True))
    scene.add_obstacle(box(-2.0, 1.0, -0.9, 1.2))
    scene.add_obstacle(box(-2.0, -1.2, -0.9, -1.0))
    neural_controller.dynamics_model.scene = scene

    # Run the experiments and plot
    rollout_experiment.run_and_plot(neural_controller, display_plots=True)
    # contour_experiment.run_and_plot(neural_controller, display_plots=True)


if __name__ == "__main__":
    eval_and_plot_linear_satellite()
