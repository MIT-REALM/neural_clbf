import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

from neural_clbf.controllers import NeuralCBFController


matplotlib.use('TkAgg')


def plot_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "saved_models/review/linear_satellite_cbf.ckpt"
    # log_file = "logs/linear_satellite_cbf/tanh/commit_7390ab2/version_0/checkpoints/epoch=20-step=3695.ckpt"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # Tweak parameters
    neural_controller.cbf_relaxation_penalty = 1e2
    neural_controller.clf_lambda = 0.1
    neural_controller.controller_period = 0.01

    # Tweak experiments
    neural_controller.experiment_suite.experiments[0].n_grid = 50  # 200
    neural_controller.experiment_suite.experiments[1].t_sim = 30.0
    neural_controller.experiment_suite.experiments[1].start_x = torch.tensor(
        [[0.5, 0.5, 0.0, -0.1, -0.1, -1.0]]
    )
    neural_controller.experiment_suite.experiments[1].other_index = [2]
    neural_controller.experiment_suite.experiments[1].other_label = ["$z$"]

    # Run the experiments and save the results
    grid_df = neural_controller.experiment_suite.experiments[0].run(neural_controller)
    traj_df = neural_controller.experiment_suite.experiments[1].run(neural_controller)

    # Plot in 3D
    sns.set_theme(context="talk", style="white")
    ax = plt.axes(projection="3d")

    # Plot the trajectory
    ax.plot3D(traj_df["$x$"], traj_df["$y$"], traj_df["$z$"], "black")
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_xticks(np.linspace(-0.75, 0.75, 2))
    ax.set_yticks(np.linspace(-0.75, 0.75, 2))
    ax.set_zticks(np.linspace(-0.75, 0.75, 2))

    # fig, axs = plt.subplots(1, 2)
    # distance = traj_df["$x$"].abs() + traj_df["$y$"].abs() + traj_df["$z$"].abs()
    # axs[0].plot(traj_df["t"], distance)
    # axs[1].plot(traj_df["t"], traj_df["V"])

    # Plot the CLF
    # contours = ax.tricontourf(
    #     grid_df["$x$"],
    #     grid_df["$y$"],
    #     grid_df["V"],
    #     cmap=sns.color_palette("rocket", as_cmap=True),
    #     alpha=0.2,
    #     levels=20,
    # )
    # plt.colorbar(contours, ax=ax, orientation="vertical")

    # ax.plot([], [], c="green", label="Safe Region")
    # ax.tricontour(
    #     grid_df["$x$"],
    #     grid_df["$y$"],
    #     grid_df["Safe region"] - 0.5,
    #     colors=["green"],
    #     levels=[0.0],
    # )
    # ax.plot([], [], c="magenta", label="Unsafe Region")
    # ax.tricontour(
    #     grid_df["$x$"],
    #     grid_df["$y$"],
    #     grid_df["Unsafe region"] - 0.5,
    #     colors=["magenta"],
    #     levels=[0.0],
    # )

    ax.plot([], [], c="blue", label="V(x) = c")
    ax.tricontour(
        grid_df["$x$"],
        grid_df["$y$"],
        grid_df["V"],
        colors=["blue"],
        levels=[0.0],
    )

    # Plot sphere
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = 0.25 * np.cos(u) * np.sin(v)
    y = 0.25 * np.sin(u) * np.sin(v)
    z = 0.25 * np.cos(v)
    ax.plot_surface(x, y, z, color="magenta", alpha=1.0, zorder=0)

    plt.show()


if __name__ == "__main__":
    plot_linear_satellite()
