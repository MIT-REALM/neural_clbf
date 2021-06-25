import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    # Beautify plots
    sns.set_theme(context="talk", style="white")
    mpc_color = sns.color_palette("pastel")[0]
    rclbf_color = sns.color_palette("pastel")[1]

    # Load the data from the CSVs
    filename = "sim_traces/stcar_rCLBF-QP_dt=0-01_omega_ref=1-5-sine.csv"
    x_rclbf = np.loadtxt(filename, delimiter=",", skiprows=1)
    filename = "sim_traces/stcar_rmpc_dt=0-1_omega_ref=1-5-sine.csv"
    x_mpc1 = np.loadtxt(filename, delimiter=",", skiprows=1)
    filename = "sim_traces/stcar_rmpc_dt=0-25_omega_ref=1-5-sine.csv"
    x_mpc25 = np.loadtxt(filename, delimiter=",", skiprows=1)

    num_timesteps = x_rclbf.shape[0]
    t_sim = 5.0

    # Extract x and y and compute the error
    xy_rclbf = x_rclbf[:, [0, 1]]
    xy_err_rclbf = np.linalg.norm(xy_rclbf, axis=-1)

    xy_mpc1 = x_mpc1[:, [0, 1]]
    xy_err_mpc1 = np.linalg.norm(xy_mpc1, axis=-1)

    xy_mpc25 = x_mpc25[:, [0, 1]]
    xy_err_mpc25 = np.linalg.norm(xy_mpc25, axis=-1)

    # Plot
    fig, ax = plt.subplots(1, 1)
    t = np.linspace(0, t_sim, num_timesteps)
    ax.plot(t, xy_err_rclbf, c=rclbf_color, label="rCLBF-QP", linewidth=5)
    ax.plot(
        t,
        xy_err_mpc1,
        c=mpc_color,
        linestyle="dashed",
        label="rMPC ($dt=0.1$)",
        linewidth=5,
    )
    ax.plot(
        t,
        xy_err_mpc25,
        c=mpc_color,
        linestyle="solid",
        label="rMPC ($dt=0.25$)",
        linewidth=5,
    )

    # Pretty plot
    ax.set_xlabel("$t$")
    ax.set_ylabel("Cartesian Tracking Error (m)")
    ax.legend(fontsize=25, loc="upper left")
    ax.set_xlim([0, t_sim])
    # ax.set_ylim([-0.5, 10])

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(25)

    fig.tight_layout()
    plt.show()

    print("STCar maximum tracking error:")
    print(f"\trCLBF-QP: {xy_err_rclbf.max()}")
    print(f"\trMPC dt0.1: {xy_err_mpc1.max()}")
    print(f"\trMPC dt0.25: {xy_err_mpc25.max()}")


if __name__ == "__main__":
    main()
