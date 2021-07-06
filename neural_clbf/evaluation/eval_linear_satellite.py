from neural_clbf.controllers import NeuralCLBFController


def eval_linear_satellite():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/aas/linear_satellite/commit_30aef5d/"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_dir + "v0.ckpt")

    # Increase the simulation time and the resolution of the grid.
    neural_controller.experiment_suite.experiments[1].t_sim = 200.0
    neural_controller.experiment_suite.experiments[0].n_grid = 500

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_save_to_csv(
        neural_controller, log_dir + "experiments"
    )


if __name__ == "__main__":
    eval_linear_satellite()
