import numpy as np
from shapely.geometry import box
import torch

from neural_clbf.controllers import NeuralObsBFController
from neural_clbf.systems.planar_lidar_system import Scene


def eval_and_plot_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/perception/turtlebot2d/commit_bcbc69e/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # The rollout experiment is the one we're interested in. It has index 1.
    rollout_experiment = neural_controller.experiment_suite.experiments[1]

    # Modify parameters
    rollout_experiment.t_sim = 5
    rollout_experiment.start_x = torch.tensor(
        [
            # [4.5, 2.5, np.pi / 2],
            # [-4.5, 2.5, np.pi / 2],
            [-3.05, 1.75, -np.pi / 4],
            # [-3.25, 1.75, 0.0],
            # [-4.5, -2.5, -np.pi / 2],
            # [4.5, -2.5, -np.pi / 2],
        ]
    )
    neural_controller.controller_period = 0.1
    neural_controller.h_alpha = 0.0
    neural_controller.lookahead_dual_penalty = 200.0
    neural_controller.lookahead_grid_n = 2
    neural_controller.dynamics_model.field_of_view = (0.0, 0.0)
    neural_controller.dynamics_model.num_rays = 1

    # Modify scene
    scene = Scene([])
    scene.add_walls(10.0)
    scene.add_obstacle(box(-3.0, 1.0, -1.0, 2.0))
    neural_controller.dynamics_model.scene = scene

    # Run the experiments and plot
    rollout_experiment.run_and_plot(neural_controller, display_plots=True)


if __name__ == "__main__":
    eval_and_plot_linear_satellite()
