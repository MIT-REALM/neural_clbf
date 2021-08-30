import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.offsetbox import (
    OffsetImage,
    AnnotationBbox,
)
from celluloid import Camera
import torch

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.experiments import CLFVerificationExperiment


def eval_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/linear_satellite_cbf/commit_318eb38/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Increase the resolution of the grid.
    neural_controller.experiment_suite.experiments[0].n_grid = 500

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_save_to_csv(
        neural_controller, log_dir + "experiments"
    )


def plot_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/linear_satellite_cbf/commit_318eb38/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Load the saved data
    results_df = pd.read_csv(log_dir + "experiments/2021-08-02_20_30_57/V Contour.csv")

    # Run the experiments and save the results
    neural_controller.experiment_suite.experiments[0].plot(
        neural_controller, results_df, display_plots=True
    )


def verify_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/linear_satellite_cbf/commit_318eb38/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Define an experiment to validate the CBF
    domain = [
        (-2.5, 2.5),
        (-2.5, 2.5),
        (-2.5, 2.5),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 1.0),
    ]
    n_grid = 10
    verification_experiment = CLFVerificationExperiment(
        "CLF_Verification", domain, n_grid=n_grid
    )

    # Run the experiments and save the results
    verification_experiment.run_and_plot(neural_controller, display_plots=True)


def animate_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/linear_satellite_cbf/commit_736760c/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v1.ckpt")
    neural_controller.cbf_relaxation_penalty = 1e5
    neural_controller.clf_lambda = 0.0

    # Save the contour experiment for later use
    contour_experiment = neural_controller.experiment_suite.experiments[0]
    contour_experiment.n_grid = 50

    # Get the dataframe from a rollout
    neural_controller.experiment_suite.experiments[1].t_sim = 20.0
    neural_controller.experiment_suite.experiments[1].start_x = torch.tensor(
        [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    rollout_df = neural_controller.experiment_suite.experiments[1].run(
        neural_controller
    )

    # Setup the plot
    fig, ax = plt.subplots()
    fig.set_size_inches((10, 10))
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.axis("off")
    ax.set_aspect("equal")
    camera = Camera(fig)

    # Draw a frame for each step of the simulation
    seconds_per_frame = 1.0 / 30
    dt = neural_controller.dynamics_model.dt
    num_frames = 0
    contour_df = pd.DataFrame()
    for i in range(0, rollout_df.shape[0], int(seconds_per_frame / dt)):
        # Add a satellite image at the origin
        ego_sat_icon = plt.imread("neural_clbf/evaluation/datafiles/ego_sat_icon.png")
        ego_imagebox = OffsetImage(ego_sat_icon, zoom=0.2)
        ego_imagebox.image.axes = ax
        ego_ab = AnnotationBbox(
            ego_imagebox,
            (0, 0),
            frameon=False,
        )
        ax.add_artist(ego_ab)

        # Add x and y gridlines
        safe_r = 0.3
        ax.plot([-5, -safe_r], [0, 0], c="k")
        ax.plot([safe_r, 5], [0, 0], c="k")
        ax.plot([0, 0], [-5, -safe_r], c="k")
        ax.plot([0, 0], [safe_r, 5], c="k")
        # Add a circle for the minimum radius
        keepout_zone = Circle((0, 0), safe_r, fill=False, ec="k")
        ax.add_artist(keepout_zone)
        # ax.text(
        #     -0.38,
        #     0.1,
        #     "Keepout Zone",
        #     rotation=45,
        #     fontsize="x-large",
        # )

        # For each timestep, plot the path of the adversary satellite
        adv_sat_icon = plt.imread("neural_clbf/evaluation/datafiles/adv_sat_icon.png")
        adv_imagebox = OffsetImage(adv_sat_icon, zoom=0.2)
        adv_imagebox.image.axes = ax
        adv_ab = AnnotationBbox(
            adv_imagebox,
            (rollout_df["$x$"][i], rollout_df["$y$"][i]),
            frameon=False,
        )
        ax.add_artist(adv_ab)

        x_so_far = rollout_df["$x$"][:i]
        y_so_far = rollout_df["$y$"][:i]
        ax.plot(x_so_far, y_so_far, c="blue")

        # Also plot a level set of the CBF at this point (only update every few frames
        # to finish in a reasonable amount of time)
        if num_frames % 5 == 0:
            current_state = torch.tensor(rollout_df["state"][i])
            contour_experiment.default_state = current_state
            contour_df = contour_experiment.run(neural_controller)

        ax.tricontour(
            contour_df["$x$"],
            contour_df["$y$"],
            contour_df["V"],
            cmap="Greys",
            levels=10,
            linewidths=1.0,
        )
        ax.tricontour(
            contour_df["$x$"],
            contour_df["$y$"],
            contour_df["V"],
            colors=["red"],
            levels=[0.0],
            linewidths=2.0,
        )

        camera.snap()
        num_frames += 1

    animation = camera.animate(interval=int(1e3 * seconds_per_frame), blit=True)
    animation.save("with_safeorbit.mp4")


if __name__ == "__main__":
    # eval_linear_satellite()
    # plot_linear_satellite()
    # verify_linear_satellite()
    animate_linear_satellite()
