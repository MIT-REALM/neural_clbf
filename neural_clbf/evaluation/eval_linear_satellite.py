import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.experiments import (
    CLFVerificationExperiment,
    RolloutNormExperiment,
)


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


def eval_linear_satellite_rollout():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/linear_satellite_cbf/commit_318eb38/"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_dir + "v0.ckpt")
    neural_controller.cbf_relaxation_penalty = float("inf")

    # Make a new experiment for a state-space rollout
    start_x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    rollout_experiment = RolloutNormExperiment(
        "Rollout",
        start_x,
        scenarios=[neural_controller.dynamics_model.nominal_params],
        n_sims_per_start=1,
        t_sim=40.0,
    )

    # Run and save
    rollout_experiment.run_and_save_to_csv(
        neural_controller, log_dir + "rollout_experiments/cbf_lqr"
    )

    # # Also try with LQR alone. Note: in my experiments, setting the relaxation penalty
    # # to zero was not sufficient. I needed to modify the source of the experiment to
    # # directly call u_nominal on the dynamics model instead of u on the cbf controller
    # neural_controller.cbf_relaxation_penalty = 0.0
    # rollout_experiment.run_and_save_to_csv(
    #     neural_controller,
    #     log_dir + "rollout_experiments/lqr"
    # )


def plot_linear_satellite_rollout():
    # Load the saved data
    log_dir = "saved_models/aas/linear_satellite_cbf/commit_318eb38/"
    lqr_results_df = pd.read_csv(log_dir + "/rollout_experiments/lqr/Rollout.csv")
    cbf_lqr_results_df = pd.read_csv(
        log_dir + "/rollout_experiments/cbf_lqr/Rollout.csv"
    )

    # Plot (manually since we need to overlay)

    # Set the color scheme
    sns.set_theme(context="talk", style="white")

    # Plot the state trajectories
    fig, rollout_ax = plt.subplots(1, 1)
    fig.set_size_inches(9, 6)

    # Plot the rollout
    rollout_ax.plot(
        lqr_results_df["t"],
        lqr_results_df["||x||"],
        color="r",
        linestyle=":",
        linewidth=3,
        label="LQR",
    )
    rollout_ax.plot(
        cbf_lqr_results_df["t"],
        cbf_lqr_results_df["||x||"],
        color="b",
        linestyle="-",
        linewidth=3,
        label="CBF + LQR",
    )

    rollout_ax.set_xlabel("$t$")
    rollout_ax.set_ylabel("$||x||$")
    for item in (
        [rollout_ax.title, rollout_ax.xaxis.label, rollout_ax.yaxis.label]
        + rollout_ax.get_xticklabels()
        + rollout_ax.get_yticklabels()
    ):
        item.set_fontsize(25)

    # Plot the unsafe boundaries
    rollout_ax.plot(
        [0, lqr_results_df.t.max()],
        [0.3, 0.3],
        color="k",
        linestyle="--",
        label="Safety constraint",
    )
    rollout_ax.plot([0, lqr_results_df.t.max()], [2.0, 2.0], color="k", linestyle="--")
    rollout_ax.legend(fontsize=25, loc="upper left")

    plt.show()


if __name__ == "__main__":
    # eval_linear_satellite()
    # plot_linear_satellite()
    # verify_linear_satellite()
    # eval_linear_satellite_rollout()
    plot_linear_satellite_rollout()
