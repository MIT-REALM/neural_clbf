import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import interpolate
import torch
import tqdm

from neural_clbf.controllers import NeuralObsBFController, ObsMPCController
from neural_clbf.experiments import (
    RolloutSuccessRateExperiment,
    ExperimentSuite,
    ObsBFVerificationExperiment,
)
import neural_clbf.evaluation.turtle2d.scenes as scene_utils
from neural_clbf.systems.planar_lidar_system import Scene


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
    log_dir = "saved_models/perception/turtlebot2d/commit_8439378/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(
        log_dir + "v0_ep72.ckpt"
    )

    # Get the experiment
    rollout_experiment = neural_controller.experiment_suite.experiments[-1]

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

    # # Also run with an MPC controller
    # mpc_controller = ObsMPCController(
    #     neural_controller.dynamics_model,
    #     neural_controller.controller_period,
    #     neural_controller.experiment_suite,
    #     neural_controller.validation_dynamics_model,
    # )
    # rollout_experiment.run_and_plot(mpc_controller, display_plots=True)


def eval_and_plot_turtlebot_random_scene():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/perception/turtlebot2d/commit_8439378/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(
        log_dir + "v0_ep72.ckpt"
    )

    # Get the experiment
    rollout_experiment = neural_controller.experiment_suite.experiments[-1]

    # Modify rollout parameters
    rollout_experiment.t_sim = 4
    neural_controller.lookahead_grid_n = 8
    neural_controller.controller_period = 0.1
    neural_controller.dynamics_model.dt = 0.01
    neural_controller.lookahead_dual_penalty = 1e3
    # neural_controller.debug_mode_exploratory = True
    # neural_controller.debug_mode_goal_seeking = True

    room_size = 10.0
    num_obstacles = 8
    box_size_range = (0.75, 1.75)
    position_range = (-4.0, 4.0)
    rotation_range = (-np.pi, np.pi)
    scene = Scene([])
    scene.add_walls(room_size)
    scene.add_random_boxes(
        num_obstacles,
        box_size_range,
        position_range,
        position_range,
        rotation_range,
    )
    neural_controller.dynamics_model.scene = scene  # type: ignore

    # Run the experiments and plot
    rollout_experiment.run_and_plot(neural_controller, display_plots=True)

    # # Also run with an MPC controller
    # mpc_controller = ObsMPCController(
    #     neural_controller.dynamics_model,
    #     neural_controller.controller_period,
    #     neural_controller.experiment_suite,
    #     neural_controller.validation_dynamics_model,
    # )
    # rollout_experiment.run_and_plot(mpc_controller, display_plots=True)


def eval_turtlebot_neural_cbf_mpc_success_rates():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/perception/turtlebot2d/commit_8439378/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(
        log_dir + "v0_ep72.ckpt"
    )

    # Make the experiment
    rollout_experiment = RolloutSuccessRateExperiment(
        "success_rate",
        "Neural oCBF/oCLF (ours)",
        n_sims=500,
        t_sim=10.0,
    )
    experiment_suite = ExperimentSuite([rollout_experiment])

    # # Run the experiments and save the results
    # experiment_suite.run_all_and_save_to_csv(
    #     neural_controller, log_dir + "experiments_neural_ocbf"
    # )

    # Also run with an MPC controller
    mpc_controller = ObsMPCController(
        neural_controller.dynamics_model,
        neural_controller.controller_period,
        neural_controller.experiment_suite,
        neural_controller.validation_dynamics_model,
    )
    rollout_experiment.algorithm_name = "MPC"
    experiment_suite.run_all_and_save_to_csv(
        mpc_controller, log_dir + "experiments_mpc_contingent"
    )

    # # Also run with a state-based controller
    # log_dir = "saved_models/perception/turtlebot2d_state/commit_f63b307/"
    # neural_state_controller = NeuralObsBFController.load_from_checkpoint(
    #     log_dir + "v0.ckpt"
    # )
    # experiment_suite.run_all_and_save_to_csv(
    #     neural_state_controller, log_dir + "experiments_neural_scbf"
    # )


def eval_and_plot_turtlebot_select_scene():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/perception/turtlebot2d/commit_8439378/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(
        log_dir + "v0_ep72.ckpt"
    )

    # Get the experiment
    rollout_experiment = neural_controller.experiment_suite.experiments[-1]

    # Modify rollout parameters
    rollout_experiment.t_sim = 10
    rollout_experiment.start_x = torch.tensor(
        [
            [-4.0, 4.0, 0.0],
        ]
    )
    # experiment_suite = ExperimentSuite([rollout_experiment])

    # Load the selected scene
    neural_controller.dynamics_model.scene = scene_utils.saved_random_scene()

    # Run the experiments and plot
    rollout_experiment.run_and_plot(neural_controller, display_plots=True)
    # experiment_suite.run_all_and_save_to_csv(
    #     neural_controller, log_dir + "experiments_neural_ocbf"
    # )

    # Also run with an MPC controller
    mpc_controller = ObsMPCController(
        neural_controller.dynamics_model,
        neural_controller.controller_period,
        neural_controller.experiment_suite,
        neural_controller.validation_dynamics_model,
    )
    rollout_experiment.run_and_plot(mpc_controller, display_plots=True)
    # experiment_suite.run_all_and_save_to_csv(
    #     mpc_controller, log_dir + "experiments_mpc_contingent"
    # )

    # # Also run with a state-based controller
    # log_dir = "saved_models/perception/turtlebot2d_state/commit_f63b307/"
    # neural_state_controller = NeuralObsBFController.load_from_checkpoint(
    #     log_dir + "v0.ckpt"
    # )
    # neural_state_controller.dynamics_model.scene = scene_utils.saved_random_scene()
    # experiment_suite.run_all_and_save_to_csv(
    #     neural_state_controller, log_dir + "experiments_neural_scbf"
    # )


def plot_select_scene():
    # Load data
    log_dir = "saved_models/perception/turtlebot2d/commit_8439378/"
    state_log_dir = "saved_models/perception/turtlebot2d_state/commit_f63b307/"

    ocbf_df = pd.read_csv(
        log_dir + "experiments_neural_ocbf/2021-09-01_17_57_56/Rollout.csv"
    )
    scbf_df = pd.read_csv(
        state_log_dir + "experiments_neural_scbf/2021-09-01_17_58_44/Rollout.csv"
    )
    mpc_df = pd.read_csv(
        log_dir + "experiments_mpc_contingent/2021-11-12_14_46_18/Rollout.csv"
    )
    ppo_df = pd.read_csv(log_dir + "experiments_ppo/2021-09-01_21_32_00/trace.csv")

    # Add the start point and smooth the ppo trace
    start = pd.DataFrame([{"$x$": -4.0, "$y$": 4.0, "$t$": 0.0}])
    ppo_df = pd.concat([start, ppo_df])

    # Set the color scheme
    sns.set_theme(context="talk", style="white")
    sns.set_style({"font.family": "serif"})

    # Create the axes
    fig, ax = plt.subplots()

    # Plot the environment
    scene_utils.saved_random_scene().plot(ax)

    ax.plot(
        [], [], color=sns.color_palette()[0], label="Observation-based CBF/CLF (ours)"
    )
    ax.plot([], [], color=sns.color_palette()[1], label="State-based CBF/CLF")
    ax.plot([], [], color=sns.color_palette()[2], label="MPC")
    ax.plot([], [], color=sns.color_palette()[3], label="PPO")

    # Plot oCBF
    ax.plot(
        ocbf_df["$x$"].to_numpy(),
        ocbf_df["$y$"].to_numpy(),
        linestyle="-",
        linewidth=5,
        color=sns.color_palette()[0],
    )

    # Plot sCBF
    ax.plot(
        scbf_df["$x$"].to_numpy(),
        scbf_df["$y$"].to_numpy(),
        linestyle="-",
        color=sns.color_palette()[1],
    )

    # Plot MPC
    ax.plot(
        mpc_df["$x$"].to_numpy(),
        mpc_df["$y$"].to_numpy(),
        linestyle="-",
        color=sns.color_palette()[2],
    )

    # Plot PPO smoothed
    ppo_t = ppo_df["$t$"].to_numpy()
    mpc_t = mpc_df["t"].to_numpy()
    ppo_x = ppo_df["$x$"].to_numpy()
    ppo_y = ppo_df["$y$"].to_numpy()
    x_smooth = interpolate.interp1d(ppo_t, ppo_x, kind="cubic")
    y_smooth = interpolate.interp1d(ppo_t, ppo_y, kind="cubic")
    ax.plot(
        x_smooth(mpc_t),
        y_smooth(mpc_t),
        linestyle=":",
        color=sns.color_palette()[3],
    )

    ax.legend(loc="lower left")
    ax.set_ylim([-2, 5.5])
    ax.set_xlim([-5.5, 3])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def validate_neural_cbf():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/perception/turtlebot2d/commit_8439378/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(
        log_dir + "v0_ep72.ckpt"
    )

    # Make the verification experiment
    verification_experiment = ObsBFVerificationExperiment("verification", 1000)

    # Increase the dual penalty so any violations of the CBF condition are clear
    neural_controller.lookahead_dual_penalty = 1e8

    # Run the experiments and save the results. Gotta do this multiple times
    # to accomodate memory
    num_infeasible = 0
    prog_bar_range = tqdm.trange(100, desc="Validating BF", leave=True)
    for i in prog_bar_range:
        df = verification_experiment.run(neural_controller)
        num_infeasible += df["# infeasible"][0]

    print(f"Total samples {100 * 1000}, # infeasible: {num_infeasible}")


if __name__ == "__main__":
    # eval_and_plot_turtlebot_room()
    # eval_and_plot_turtlebot_bugtrap()
    # eval_and_plot_turtlebot_training()
    eval_and_plot_turtlebot_random_scene()
    # eval_turtlebot_neural_cbf_mpc_success_rates()
    # eval_and_plot_turtlebot_select_scene()
    # plot_select_scene()
    # validate_neural_cbf()
