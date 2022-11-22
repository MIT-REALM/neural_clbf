from copy import copy
import matplotlib

from neural_clbf.controllers import NeuralCLBFController


matplotlib.use('TkAgg')


def plot_stcar_trajectory():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "saved_models/review/stcar_clf.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Tweak controller params; boost relaxation penalty ultra-high to prevent
    # unnecessary relaxation.
    neural_controller.clf_relaxation_penalty = 1e8

    # Decrease the disturbance so we test on a subset of the training disturbance
    scenarios = []
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    omega_ref_vals = [-0.5, 0.5]
    for omega_ref in omega_ref_vals:
        s = copy(nominal_params)
        s["omega_ref"] = omega_ref

        scenarios.append(s)

    neural_controller.scenarios = scenarios
    neural_controller.controller_period = 0.01

    # Tweak experiment params
    neural_controller.experiment_suite.experiments[1].t_sim = 5.0

    # Run the experiments and save the results
    neural_controller.experiment_suite.experiments[1].run_and_plot(
        neural_controller, display_plots=True
    )


def plot_stcar_clf():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_file = "saved_models/review/stcar_clf.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # Tweak controller params; boost relaxation penalty ultra-high to prevent
    # unnecessary relaxation.
    neural_controller.clf_relaxation_penalty = 1e8

    # Decrease the disturbance so we test on a subset of the training disturbance
    scenarios = []
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    omega_ref_vals = [-0.5, 0.5]
    for omega_ref in omega_ref_vals:
        s = copy(nominal_params)
        s["omega_ref"] = omega_ref

        scenarios.append(s)

    neural_controller.scenarios = scenarios
    neural_controller.controller_period = 0.01

    # Tweak experiment params
    neural_controller.experiment_suite.experiments[0].n_grid = 20

    # Run the experiments and save the results
    neural_controller.experiment_suite.experiments[0].run_and_plot(
        neural_controller, display_plots=True
    )


if __name__ == "__main__":
    # plot_stcar_trajectory()
    plot_stcar_clf()
