from copy import copy
import torch

from neural_clbf.systems import KSCar
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)

# Import the plotting callbacks, which seem to be needed to load from the checkpoint
from neural_clbf.experiments.train_single_track_car import (  # noqa
    rollout_plotting_cb,  # noqa
    clbf_plotting_cb,  # noqa
)

from neural_clbf.experiments.data_generation.kscar_s_curve_rollout import (
    save_kscar_s_curve_rollout,
)


def doMain():
    checkpoint_file = "saved_models/good/kscar/e6f766a_v1.ckpt"

    controller_period = 0.01
    simulation_dt = 0.001

    # Define the dynamics model
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    kscar = KSCar(nominal_params, dt=simulation_dt, controller_dt=controller_period)

    # Initialize the DataModule
    initial_conditions = [
        (-0.1, 0.1),  # sxe
        (-0.1, 0.1),  # sye
        (-0.1, 0.1),  # delta
        (-0.1, 0.1),  # ve
        (-0.1, 0.1),  # psi_e
    ]

    # Define the scenarios
    scenarios = []
    omega_ref_vals = [-1.5, 1.5]
    for omega_ref in omega_ref_vals:
        s = copy(nominal_params)
        s["omega_ref"] = omega_ref

        scenarios.append(s)

    data_module = EpisodicDataModule(
        kscar,
        initial_conditions,
        trajectories_per_episode=1,
        trajectory_length=10,
        fixed_samples=100,
        max_points=5000000,
        val_split=0.1,
        batch_size=64,
    )

    clbf_controller = NeuralCLBFController.load_from_checkpoint(
        checkpoint_file,
        map_location=torch.device("cpu"),
        dynamics_model=kscar,
        scenarios=scenarios,
        datamodule=data_module,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        u_nn_hidden_layers=2,
        u_nn_hidden_size=64,
        clbf_lambda=1.0,
        safety_level=1.0,
        controller_period=controller_period,
        clbf_relaxation_penalty=1e1,
        primal_learning_rate=1e-3,
        penalty_scheduling_rate=0,
        num_init_epochs=11,
        optimizer_alternate_epochs=1,
        epochs_per_episode=200,
        use_nominal_in_qp=False,
    )

    save_kscar_s_curve_rollout(
        clbf_controller, "rCLBF-QP", controller_period, kscar, randomize_path=True
    )


if __name__ == "__main__":
    doMain()
