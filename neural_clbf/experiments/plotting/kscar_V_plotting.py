from copy import copy
import torch
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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


@torch.no_grad()
def doMain():
    sns.set_theme(context="talk", style="white")

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

    clbf_net = NeuralCLBFController.load_from_checkpoint(
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

    domain = [(-2.0, 2.0), (-2.0, 2.0)]
    n_grid = 100
    x_axis_index = KSCar.SXE
    y_axis_index = KSCar.SYE
    x_axis_label = "$x - x_{ref}$"
    y_axis_label = "$y - y_{ref}$"

    x_vals = torch.linspace(domain[0][0], domain[0][1], n_grid, device=clbf_net.device)
    y_vals = torch.linspace(domain[1][0], domain[1][1], n_grid, device=clbf_net.device)

    # Set up tensors to store the results
    V_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    relax_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    unsafe_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)
    safe_grid = torch.zeros(n_grid, n_grid).type_as(x_vals)

    default_state = torch.zeros(1, clbf_net.dynamics_model.n_dims).type_as(x_vals)

    # Make a copy of the default state, which we'll modify on every loop
    x = default_state.clone().detach().reshape(1, clbf_net.dynamics_model.n_dims)
    prog_bar_range = tqdm.trange(n_grid, desc="Plotting CLBF", leave=True)
    print("Plotting CLBF on grid...")
    for i in prog_bar_range:
        for j in range(n_grid):
            # Adjust x to be at the current grid point
            x[0, x_axis_index] = x_vals[i]
            x[0, y_axis_index] = y_vals[j]

            # Get the value of the CLBF
            V_grid[j, i] = clbf_net.V(x)

            # Get the QP relaxation for all points where V < safe_level
            # if V_grid[j, i] <= clbf_net.safe_level:
            _, r, _ = clbf_net.solve_CLBF_QP(x)  # type: ignore
            relax_grid[j, i] = r.max()

            # Get the goal, safe, or unsafe classification
            if clbf_net.dynamics_model.safe_mask(x).all():
                safe_grid[j, i] = 1
            elif clbf_net.dynamics_model.unsafe_mask(x).all():
                unsafe_grid[j, i] = 1

    # Make the plots
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)

    # First plot V
    contours = axs.contourf(
        x_vals.cpu(), y_vals.cpu(), V_grid.cpu(), cmap="magma", levels=20
    )
    plt.colorbar(contours, ax=axs, orientation="horizontal")
    # Plot safe levels
    # axs.plot([], [], c="g", label="Safe")
    # axs.plot([], [], c="r", label="Unsafe")
    axs.plot([], [], c="blue", label="V(x) = c")
    safe_level = clbf_net.safe_level
    if isinstance(safe_level, torch.Tensor):
        safe_level = safe_level.item()
    # axs.contour(
    #     x_vals.cpu(),
    #     y_vals.cpu(),
    #     unsafe_grid.cpu(),
    #     colors=["red"],
    #     levels=[0.5],  # type: ignore
    # )
    # axs.contour(
    #     x_vals.cpu(),
    #     y_vals.cpu(),
    #     safe_grid.cpu(),
    #     colors=["green"],
    #     levels=[0.5],  # type: ignore
    # )
    # And unsafe levels
    unsafe_level = clbf_net.unsafe_level
    if isinstance(unsafe_level, torch.Tensor):
        unsafe_level = unsafe_level.item()
    axs.contour(
        x_vals.cpu(),
        y_vals.cpu(),
        V_grid.cpu(),
        colors=["blue"],
        levels=[unsafe_level],  # type: ignore
    )

    # Plot the relaxation
    relax_grid[relax_grid <= 0] = -1
    contours = axs.contour(
        x_vals.cpu(), y_vals.cpu(), relax_grid, colors=["white"], levels=[0.0]
    )

    axs.set_xlabel(x_axis_label)
    axs.set_ylabel(y_axis_label)
    axs.set_title("$V$")
    axs.legend(loc="upper right", fontsize=25)
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    doMain()
