import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Beautify plots
sns.set_theme(context="talk", style="white")
obs_color = sns.color_palette("pastel")[3]
mpc_ecolor = sns.color_palette("pastel")[0] + (1.0,)
mpc_fcolor = sns.color_palette("pastel")[0] + (0.0,)
rclbf_color = sns.color_palette("pastel")[1]

# Load the data from the CSVs
filename = "sim_traces/quad3d_rCLBF-QP_dt=0-001_m=1-0.csv"
x_rclbf_low = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rCLBF-QP_dt=0-001_m=1-5.csv"
x_rclbf_high = np.loadtxt(filename, delimiter=",", skiprows=1)

filename = "sim_traces/quad3d_rmpc_dt=0-1_m=1-0.csv"
x_mpc1_low = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-1_m=1-5.csv"
x_mpc1_high = np.loadtxt(filename, delimiter=",", skiprows=1)

filename = "sim_traces/quad3d_rmpc_dt=0-1_m=1-0.csv"
x_mpc1_0 = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-1_m=1-1.csv"
x_mpc1_1 = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-1_m=1-2.csv"
x_mpc1_2 = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-1_m=1-3.csv"
x_mpc1_3 = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-1_m=1-4.csv"
x_mpc1_4 = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-1_m=1-5.csv"
x_mpc1_5 = np.loadtxt(filename, delimiter=",", skiprows=1)

filename = "sim_traces/quad3d_rmpc_dt=0-25_m=1-0.csv"
x_mpc25_0 = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-25_m=1-1.csv"
x_mpc25_1 = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-25_m=1-2.csv"
x_mpc25_2 = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-25_m=1-3.csv"
x_mpc25_3 = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-25_m=1-4.csv"
x_mpc25_4 = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad3d_rmpc_dt=0-25_m=1-5.csv"
x_mpc25_5 = np.loadtxt(filename, delimiter=",", skiprows=1)

num_timesteps = x_rclbf_low.shape[0]
t_sim = 10

# Plot
fig, ax = plt.subplots(1, 1)
t = np.linspace(0, t_sim, num_timesteps)
# ax.plot([], c=rclbf_color, label="rCLBF-QP")
# ax.plot([], c=mpc_ecolor, linestyle="dashed", label="rMPC ($dt=0.1$)")
# ax.plot([], c=mpc_ecolor, linestyle="solid", label="rMPC ($dt=0.25$)")

ax.fill_between(
    t,
    -x_rclbf_low[:, 2],
    -x_rclbf_high[:, 2],
    color=rclbf_color,
    alpha=0.9,
    label="rCLBF-QP",
)

z_mpc1 = np.vstack(
    [
        x_mpc1_0[:, 2],
        x_mpc1_1[:, 2],
        x_mpc1_2[:, 2],
        x_mpc1_3[:, 2],
        x_mpc1_4[:, 2],
        x_mpc1_5[:, 2],
    ]
)
min_trace = np.min(z_mpc1, axis=0)
max_trace = np.max(z_mpc1, axis=0)
ax.fill_between(
    t[:4999],
    min_trace[:4999],
    max_trace[:4999],
    ec=mpc_ecolor,
    fc=mpc_fcolor,
    hatch="///",
    lw=3.0,
    label="rMPC ($dt=0.1$)",
)

z_mpc25 = np.vstack(
    [
        x_mpc25_0[:, 2],
        x_mpc25_1[:, 2],
        x_mpc25_2[:, 2],
        x_mpc25_3[:, 2],
        x_mpc25_4[:, 2],
        x_mpc25_5[:, 2],
    ]
)
min_trace = np.min(z_mpc25, axis=0)
max_trace = np.max(z_mpc25, axis=0)
ax.fill_between(
    t[:4999],
    min_trace[:4999],
    max_trace[:4999],
    color=mpc_ecolor,
    alpha=0.9,
    label="rMPC ($dt=0.25$)",
)

ax.plot(t, t * 0.0 - 0.0, c="g")
ax.text(0.1, 0.05 - 0.0, "Safe", fontsize=25)
ax.plot(t, t * 0.0 - 0.3, c="r")
ax.text(0.1, -0.25 - 0.3, "Unsafe", fontsize=25)

# Pretty plot
ax.set_xlabel("$t$")
ax.set_ylabel("$z$")
ax.legend(fontsize=25, loc="upper right")
ax.set_ylim([-0.7, 3])
ax.set_xlim([0, 5.0])

for item in (
    [ax.title, ax.xaxis.label, ax.yaxis.label]
    + ax.get_xticklabels()
    + ax.get_yticklabels()
):
    item.set_fontsize(25)

fig.tight_layout()
plt.show()
