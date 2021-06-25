import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    # Beautify plots
    sns.set_theme(context="talk", style="white")
    mpc_ecolor = sns.color_palette("pastel")[0] + (1.0,)
    mpc_fcolor = sns.color_palette("pastel")[0] + (0.0,)
    rclbf_color = sns.color_palette("pastel")[1]

    # Load the data from the CSVs
    filename = "sim_traces/neural_lander_rCLBF-QP_dt=0-001_m=1-47.csv"
    x_rclbf_low = np.loadtxt(filename, delimiter=",", skiprows=1)
    filename = "sim_traces/neural_lander_rCLBF-QP_dt=0-001_m=2-0.csv"
    x_rclbf_high = np.loadtxt(filename, delimiter=",", skiprows=1)

    filename = "sim_traces/neural_lander_rmpc_dt=0-1_m=1-47.csv"
    x_mpc1_low = np.loadtxt(filename, delimiter=",", skiprows=1)
    filename = "sim_traces/neural_lander_rmpc_dt=0-1_m=2-0.csv"
    x_mpc1_high = np.loadtxt(filename, delimiter=",", skiprows=1)

    filename = "sim_traces/neural_lander_rmpc_dt=0-25_m=1-47.csv"
    x_mpc25_low = np.loadtxt(filename, delimiter=",", skiprows=1)
    filename = "sim_traces/neural_lander_rmpc_dt=0-25_m=2-0.csv"
    x_mpc25_high = np.loadtxt(filename, delimiter=",", skiprows=1)

    num_timesteps = x_rclbf_low.shape[0]
    t_sim = 5

    # z_mpc1 = np.vstack(
    #     [
    #         x_mpc1_low[:, 2],
    #         x_mpc1_high[:, 2],
    #     ]
    # )
    # z_mpc25 = np.vstack(
    #     [
    #         x_mpc25_low[:, 2],
    #         x_mpc25_high[:, 2],
    #     ]
    # )

    # Plot
    fig, ax = plt.subplots(1, 1)
    t = np.linspace(0, t_sim, num_timesteps)
    # ax.plot([], c=rclbf_color, label="rCLBF-QP")
    # ax.plot([], c=mpc_ecolor, linestyle="dashed", label="rMPC ($dt=0.1$)")
    # ax.plot([], c=mpc_ecolor, linestyle="solid", label="rMPC ($dt=0.25$)")

    ax.fill_between(
        t,
        x_rclbf_low[:, 2],
        x_rclbf_high[:, 2],
        color=rclbf_color,
        alpha=0.9,
        label="rCLBF-QP",
    )
    ax.fill_between(
        t[:4999],
        x_mpc1_low[:4999, 2],
        x_mpc1_high[:4999, 2],
        ec=mpc_ecolor,
        fc=mpc_fcolor,
        hatch="///",
        lw=3.0,
        label="rMPC ($dt=0.1$)",
    )

    ax.fill_between(
        t[:4999],
        x_mpc25_low[:4999, 2],
        x_mpc25_high[:4999, 2],
        color=mpc_ecolor,
        alpha=0.5,
        label="rMPC ($dt=0.25$)",
    )

    ax.plot(t, t * 0.0 - 0.05, c="g")
    ax.text(2.1, 0.0 - 0.0, "Safe", fontsize=25)
    ax.plot(t, t * 0.0 - 0.3, c="r")
    ax.text(2.1, -0.25 - 0.2, "Unsafe", fontsize=25)

    # Pretty plot
    ax.set_xlabel("$t$")
    ax.set_ylabel("$z$")
    ax.legend(fontsize=25, loc="upper left")
    ax.set_ylim([-0.7, 1.5])
    ax.set_xlim([0, 5.0])

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(25)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
