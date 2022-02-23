import os
import sys
import inspect
import random

import torch

from math import pi


# Add the parent directory to the path to load the trainer module
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))  # type: ignore
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from trainer import Trainer  # noqa
from dynamics import (  # noqa
    f_damped_integrator,
    AB_damped_integrator,
    f_turtlebot,
    AB_turtlebot,
)
from nonlinear_mpc_controller import turtlebot_mpc_casadi  # noqa


def test_trainer_init():
    """Test initializing the trainer object; also returns a trainer object for
    use in other tests."""

    # Create a new trainer object, and make sure it works
    hyperparameters = {
        "n_state_dims": 2,
        "n_control_dims": 1,
        "lambda_M": 0.1,
        "metric_hidden_layers": 2,
        "metric_hidden_units": 32,
        "policy_hidden_layers": 2,
        "policy_hidden_units": 32,
        "learning_rate": 1e-3,
        "batch_size": 10,
        "n_trajs": 10,
        "controller_dt": 0.1,
        "sim_dt": 1e-2,
        "demonstration_noise": 0.3,
    }
    state_space = [
        (-5.0, 5.0),  # px
        (-5.0, 5.0),  # vx
    ]
    error_bounds = [
        5.0,  # px
        5.0,  # vx
    ]
    control_bounds = [
        3.0,  # u
    ]

    expert_horizon = 1.0

    def dummy_expert(x, x_ref, u_ref):
        return u_ref[0, :]

    my_trainer = Trainer(
        "test_network",
        hyperparameters,
        f_damped_integrator,
        AB_damped_integrator,
        dummy_expert,
        expert_horizon,
        state_space,
        error_bounds,
        control_bounds,
        0.1,  # validation_split
    )
    assert my_trainer

    return my_trainer


def test_trainer_normalize_state():
    """Test state normalization"""
    my_trainer = test_trainer_init()

    test_x = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    expected_x_norm = test_x / 5.0

    x_norm = my_trainer.normalize_state(test_x)
    assert torch.allclose(x_norm, expected_x_norm)


def test_trainer_normalize_error():
    """Test state normalization"""
    # Set a random seed for repeatability
    random.seed(0)
    torch.manual_seed(0)

    my_trainer = test_trainer_init()

    test_x_err = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    expected_x_err_norm = test_x_err / 5.0

    x_err_norm = my_trainer.normalize_error(test_x_err)
    assert torch.allclose(x_err_norm, expected_x_err_norm)


def test_trainer_positive_definite_loss():
    """Test loss"""
    my_trainer = test_trainer_init()

    # Define a generic matrix known to be positive definite
    test_M = torch.tensor(
        [
            [1.0, 0.1, 0.0],
            [0.1, 1.0, -0.1],
            [0.0, -0.1, 1.0],
        ]
    ).reshape(-1, 3, 3)

    pd_loss = my_trainer.positive_definite_loss(test_M)

    assert (pd_loss <= torch.tensor(0.0)).all()

    # Now repeat for a matrix known to be negative definite
    pd_loss = my_trainer.positive_definite_loss(-1.0 * test_M)

    assert (pd_loss > torch.tensor(0.0)).all()


def test_add_data_turtlebot():
    hyperparameters = {
        "n_state_dims": 3,
        "n_control_dims": 2,
        "lambda_M": 0.1,
        "metric_hidden_layers": 2,
        "metric_hidden_units": 32,
        "policy_hidden_layers": 2,
        "policy_hidden_units": 32,
        "learning_rate": 1e-3,
        "batch_size": 10,
        "n_trajs": 5,
        "controller_dt": 0.1,
        "sim_dt": 1e-2,
        "demonstration_noise": 0.3,
    }
    state_space = [
        (-5.0, 5.0),  # px
        (-5.0, 5.0),  # py
        (-2 * pi, 2 * pi),  # theta
    ]
    error_bounds = [
        0.5,  # px
        0.5,  # py
        1.0,  # theta
    ]
    control_bounds = [
        3.0,  # v
        pi,  # omega
    ]

    expert_horizon = 1.0
    # expert_horizon = hyperparameters["controller_dt"]

    def expert(x, x_ref, u_ref):
        return turtlebot_mpc_casadi(
            x, x_ref, u_ref, hyperparameters["controller_dt"], control_bounds
        )

    my_trainer = Trainer(
        "test_trainer",
        hyperparameters,
        f_turtlebot,
        AB_turtlebot,
        expert,
        expert_horizon,
        state_space,
        error_bounds,
        control_bounds,
        0.2,  # validation_split
    )

    # Get initial amounts of data
    n_training_points_initial = my_trainer.x_training.shape[0]
    n_validation_points_initial = my_trainer.x_validation.shape[0]

    # Try adding data
    counterexample_x = torch.zeros((1, hyperparameters["n_state_dims"]))
    counterexample_x_ref = torch.zeros((1, hyperparameters["n_state_dims"]))
    counterexample_u_ref = torch.zeros((1, hyperparameters["n_control_dims"]))
    my_trainer.add_data(
        counterexample_x, counterexample_x_ref, counterexample_u_ref, 10, 5, 0.2
    )

    # Make sure we added some data
    assert my_trainer.x_training.shape[0] > n_training_points_initial
    assert my_trainer.x_validation.shape[0] > n_validation_points_initial


# Define a function to run training and save the results
# (don't run when we run pytest, only when we run this file)
def do_training_turtlebot():
    hyperparameters = {
        "n_state_dims": 3,
        "n_control_dims": 2,
        "lambda_M": 0.1,
        "metric_hidden_layers": 2,
        "metric_hidden_units": 32,
        "policy_hidden_layers": 2,
        "policy_hidden_units": 32,
        "learning_rate": 1e-3,
        "batch_size": 100,
        "n_trajs": 100,
        "controller_dt": 0.1,
        "sim_dt": 1e-2,
        "demonstration_noise": 0.3,
    }
    state_space = [
        (-5.0, 5.0),  # px
        (-5.0, 5.0),  # py
        (-2 * pi, 2 * pi),  # theta
    ]
    error_bounds = [
        0.5,  # px
        0.5,  # py
        1.0,  # theta
    ]
    control_bounds = [
        3.0,  # v
        pi,  # omega
    ]

    expert_horizon = 1.0
    # expert_horizon = hyperparameters["controller_dt"]

    def expert(x, x_ref, u_ref):
        return turtlebot_mpc_casadi(
            x, x_ref, u_ref, hyperparameters["controller_dt"], control_bounds
        )

    my_trainer = Trainer(
        (
            "clone_M_cond_2x32_policy_2x32_metric_1e4_noisy_examples_100x0.1"
            "_no_L_lr1e-3_1s_horizon"
        ),
        hyperparameters,
        f_turtlebot,
        AB_turtlebot,
        expert,
        expert_horizon,
        state_space,
        error_bounds,
        control_bounds,
        0.3,  # validation_split
    )

    n_steps = 502
    my_trainer.run_training(n_steps, debug=True, sim_every_n_steps=100)


if __name__ == "__main__":
    do_training_turtlebot()
