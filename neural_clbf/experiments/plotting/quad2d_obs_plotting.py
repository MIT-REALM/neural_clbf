import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import seaborn as sns


# Beautify plots
sns.set_theme(context="talk", style="white")
obs_color = sns.color_palette("pastel")[3]
mpc_color = sns.color_palette("pastel")[0]
mpc_ecolor = sns.color_palette("pastel")[0] + (1.0,)
mpc_fcolor = sns.color_palette("pastel")[0] + (0.0,)
rclbf_color = sns.color_palette("pastel")[1]

# Load the data from the CSVs
filename = "sim_traces/quad2d_obs_rCLBF-QP_dt=0-001_m=1-0_I=0-01.csv"
x_rclbf_low = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad2d_obs_rCLBF-QP_dt=0-001_m=1-05_I=0-0105.csv"
x_rclbf_high = np.loadtxt(filename, delimiter=",", skiprows=1)

filename = "sim_traces/quad2d_obs_rmpc_dt=0-1_m=1-0_I=0-01.csv"
x_mpc1_low = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad2d_obs_rmpc_dt=0-1_m=1-05_I=0-0105.csv"
x_mpc1_high = np.loadtxt(filename, delimiter=",", skiprows=1)

filename = "sim_traces/quad2d_obs_rmpc_dt=0-25_m=1-0_I=0-01.csv"
x_mpc25_low = np.loadtxt(filename, delimiter=",", skiprows=1)
filename = "sim_traces/quad2d_obs_rmpc_dt=0-25_m=1-05_I=0-0105.csv"
x_mpc25_high = np.loadtxt(filename, delimiter=",", skiprows=1)

num_timesteps = x_rclbf_low.shape[0]
t_sim = 5

# Plot
fig, axs = plt.subplots(1, 2)
t = np.linspace(0, t_sim, num_timesteps)

rclbf_low_goal_time = int(
    np.argmax(np.linalg.norm(x_rclbf_low[:, [0, 1]], axis=-1) <= 0.3) - 1
)
mpc1_low_goal_time = int(
    np.argmax(np.linalg.norm(x_mpc1_low[:, [0, 1]], axis=-1) <= 0.3) - 1
)
mpc25_low_goal_time = int(
    np.argmax(np.linalg.norm(x_mpc25_low[:, [0, 1]], axis=-1) <= 0.3) - 1
)
rclbf_high_goal_time = int(
    np.argmax(np.linalg.norm(x_rclbf_high[:, [0, 1]], axis=-1) <= 0.3) - 1
)
mpc1_high_goal_time = int(
    np.argmax(np.linalg.norm(x_mpc1_high[:, [0, 1]], axis=-1) <= 0.3) - 1
)
mpc25_high_goal_time = int(
    np.argmax(np.linalg.norm(x_mpc25_high[:, [0, 1]], axis=-1) <= 0.3) - 1
)

ax1 = axs[0]
ax1.plot([], c=rclbf_color, label="rCLBF-QP", linewidth=6)
ax1.plot([], c=mpc_color, label="rMPC ($dt=0.1$)", linewidth=6, linestyle="dashed")
ax1.plot([], c=mpc_color, label="rMPC ($dt=0.25$)", linewidth=1, linestyle="solid")
ax1.plot(
    x_rclbf_low[:rclbf_low_goal_time, 0],
    x_rclbf_low[:rclbf_low_goal_time, 1],
    c=rclbf_color,
    linewidth=6,
)
ax1.plot(
    x_mpc1_low[:mpc1_low_goal_time, 0],
    x_mpc1_low[:mpc1_low_goal_time, 1],
    c=mpc_color,
    linewidth=6,
    linestyle="dashed",
)
ax1.plot(
    x_mpc25_low[:mpc25_low_goal_time, 0],
    x_mpc25_low[:mpc25_low_goal_time, 1],
    c=mpc_color,
    linewidth=1,
    linestyle="solid",
)
ax1.scatter(
    [], [], label="Goal", s=1000, facecolors="none", edgecolors="k", linestyle="--"
)

# Add patches for unsafe region
obs1 = patches.Rectangle(
    (-1.0, -0.4),
    0.5,
    0.9,
    linewidth=1,
    edgecolor="r",
    facecolor=obs_color,
    label="Unsafe Region",
)
obs2 = patches.Rectangle(
    (0.0, 0.8), 1.0, 0.6, linewidth=1, edgecolor="r", facecolor=obs_color
)
ground = patches.Rectangle(
    (-4.0, -4.0), 8.0, 3.7, linewidth=1, edgecolor="r", facecolor=obs_color
)
goal = patches.Circle(
    (0.0, 0.0),
    radius=0.3,
    linewidth=2,
    edgecolor="k",
    linestyle="dashed",
    facecolor=(1.0, 1.0, 1.0, 0.0),
)
ax1.add_patch(obs1)
ax1.add_patch(obs2)
ax1.add_patch(ground)
ax1.add_patch(goal)

ax1.set_xlabel("$x$")
ax1.set_ylabel("$z$")
ax1.legend(fontsize=25, loc="upper left")
ax1.set_xlim([-2.0, 1.0])
ax1.set_ylim([-0.5, 1.5])

ax1.text(-1.5, -0.45, "$m=1.00, I=1.00\\times10^{-2}$", fontsize=25)

for item in (
    [ax1.title, ax1.xaxis.label, ax1.yaxis.label]
    + ax1.get_xticklabels()
    + ax1.get_yticklabels()
):
    item.set_fontsize(25)

ax2 = axs[1]
ax2.plot([], c=rclbf_color, label="rCLBF-QP", linewidth=6)
ax2.plot([], c=mpc_color, label="rMPC ($dt=0.1$)", linewidth=6, linestyle="dashed")
ax2.plot([], c=mpc_color, label="rMPC ($dt=0.25$)", linewidth=1, linestyle="solid")
ax2.plot(
    x_rclbf_high[:rclbf_high_goal_time, 0],
    x_rclbf_high[:rclbf_high_goal_time, 1],
    c=rclbf_color,
    linewidth=6,
)
ax2.plot(
    x_mpc1_high[:mpc1_high_goal_time, 0],
    x_mpc1_high[:mpc1_high_goal_time, 1],
    c=mpc_color,
    linewidth=6,
    linestyle="dashed",
)
ax2.plot(
    x_mpc25_high[:mpc25_high_goal_time, 0],
    x_mpc25_high[:mpc25_high_goal_time, 1],
    c=mpc_color,
    linewidth=1,
    linestyle="solid",
)
ax2.scatter(
    [], [], label="Goal", s=1000, facecolors="none", edgecolors="k", linestyle="--"
)

# Add patches for unsafe region
obs1 = patches.Rectangle(
    (-1.0, -0.4),
    0.5,
    0.9,
    linewidth=1,
    edgecolor="r",
    facecolor=obs_color,
    label="Unsafe Region",
)
obs2 = patches.Rectangle(
    (0.0, 0.8), 1.0, 0.6, linewidth=1, edgecolor="r", facecolor=obs_color
)
ground = patches.Rectangle(
    (-4.0, -4.0), 8.0, 3.7, linewidth=1, edgecolor="r", facecolor=obs_color
)
goal = patches.Circle(
    (0.0, 0.0),
    radius=0.3,
    linewidth=2,
    edgecolor="k",
    linestyle="dashed",
    facecolor=(1.0, 1.0, 1.0, 0.0),
)
ax2.add_patch(obs1)
ax2.add_patch(obs2)
ax2.add_patch(ground)
ax2.add_patch(goal)

ax2.set_xlabel("$x$")
ax2.set_ylabel("$z$")
ax2.legend(fontsize=25, loc="upper left")
ax2.set_xlim([-2.0, 1.0])
ax2.set_ylim([-0.5, 1.5])

ax2.text(-1.5, -0.45, "$m=1.05, I=1.05\\times10^{-2}$", fontsize=25)

for item in (
    [ax2.title, ax2.xaxis.label, ax2.yaxis.label]
    + ax2.get_xticklabels()
    + ax2.get_yticklabels()
):
    item.set_fontsize(25)

# fig.tight_layout()
plt.show()

# Animate the neural controller
fig, ax = plt.subplots(figsize=(18, 15))
# Add patches for unsafe region
obs1 = patches.Rectangle(
    (-1.0, -0.4),
    0.5,
    0.9,
    linewidth=1,
    edgecolor="r",
    facecolor=obs_color,
    label="Unsafe Region",
)
obs2 = patches.Rectangle(
    (0.0, 0.8), 1.0, 0.6, linewidth=1, edgecolor="r", facecolor=obs_color
)
ground = patches.Rectangle(
    (-4.0, -4.0), 8.0, 3.7, linewidth=1, edgecolor="r", facecolor=obs_color
)
ax.add_patch(obs1)
ax.add_patch(obs2)
ax.add_patch(ground)

ax.set_xlabel("$x$")
ax.set_ylabel("$z$")
ax.set_xlim([-2.0, 1.0])
ax.set_ylim([-0.5, 1.5])

quad_clbf = patches.Rectangle(
    (0.0, 0.0),
    0.1,
    0.05,
    linewidth=1,
    facecolor=rclbf_color,
    edgecolor=rclbf_color,
    label="rCLBF",
)
ax.add_patch(quad_clbf)
quad_mpc1 = patches.Rectangle(
    (0.0, 0.0),
    0.1,
    0.05,
    linewidth=1,
    ec=mpc_ecolor,
    fc=mpc_fcolor,
    hatch="///",
    label="rMPC ($dt=0.1$)",
)
ax.add_patch(quad_mpc1)
quad_mpc25 = patches.Rectangle(
    (0.0, 0.0),
    0.1,
    0.05,
    linewidth=1,
    facecolor=mpc_color,
    edgecolor=mpc_color,
    label="rMPC ($dt=0.25$)",
)
ax.add_patch(quad_mpc25)
ax.legend(fontsize=25, loc="upper left")


def animate_low(i):
    # i is the frame. At 30 fps, t = i/30
    delta_t = t_sim / num_timesteps
    t_index = int((i / 30) / delta_t)

    t_index_clbf = min(t_index, rclbf_low_goal_time)
    t_index_mpc1 = min(t_index, mpc1_low_goal_time)
    t_index_mpc25 = min(t_index, mpc25_low_goal_time)

    quad_clbf.set_xy([x_rclbf_low[t_index_clbf, 0], x_rclbf_low[t_index_clbf, 1]])
    quad_clbf._angle = -np.rad2deg(x_rclbf_low[t_index_clbf, 2])
    quad_mpc1.set_xy([x_mpc1_low[t_index_mpc1, 0], x_mpc1_low[t_index_mpc1, 1]])
    quad_mpc1._angle = -np.rad2deg(x_mpc1_low[t_index_mpc1, 2])
    quad_mpc25.set_xy([x_mpc25_low[t_index_mpc25, 0], x_mpc25_low[t_index_mpc25, 1]])
    quad_mpc25._angle = -np.rad2deg(x_mpc25_low[t_index_mpc25, 2])
    return quad_clbf, quad_mpc1, quad_mpc25


mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
anim = FuncAnimation(fig, animate_low, interval=1000 / 30, frames=int(7.5 * 30))
# anim.save('/home/cbd/Downloads/pvtol_obs_low.mov')
plt.show()
plt.draw()


fig, ax = plt.subplots(figsize=(18, 15))
# Add patches for unsafe region
obs1 = patches.Rectangle(
    (-1.0, -0.4),
    0.5,
    0.9,
    linewidth=1,
    edgecolor="r",
    facecolor=obs_color,
    label="Unsafe Region",
)
obs2 = patches.Rectangle(
    (0.0, 0.8), 1.0, 0.6, linewidth=1, edgecolor="r", facecolor=obs_color
)
ground = patches.Rectangle(
    (-4.0, -4.0), 8.0, 3.7, linewidth=1, edgecolor="r", facecolor=obs_color
)
ax.add_patch(obs1)
ax.add_patch(obs2)
ax.add_patch(ground)

ax.set_xlabel("$x$")
ax.set_ylabel("$z$")
ax.set_xlim([-2.0, 1.0])
ax.set_ylim([-0.5, 1.5])

quad_clbf = patches.Rectangle(
    (0.0, 0.0),
    0.1,
    0.05,
    linewidth=1,
    facecolor=rclbf_color,
    edgecolor=rclbf_color,
    label="rCLBF",
)
ax.add_patch(quad_clbf)
quad_mpc1 = patches.Rectangle(
    (0.0, 0.0),
    0.1,
    0.05,
    linewidth=1,
    ec=mpc_ecolor,
    fc=mpc_fcolor,
    hatch="///",
    label="rMPC ($dt=0.1$)",
)
ax.add_patch(quad_mpc1)
quad_mpc25 = patches.Rectangle(
    (0.0, 0.0),
    0.1,
    0.05,
    linewidth=1,
    facecolor=mpc_color,
    edgecolor=mpc_color,
    label="rMPC ($dt=0.25$)",
)
ax.add_patch(quad_mpc25)
ax.legend(fontsize=25, loc="upper left")


def animate_high(i):
    # i is the frame. At 30 fps, t = i/30
    delta_t = t_sim / num_timesteps
    t_index = int((i / 30) / delta_t)

    t_index_clbf = min(t_index, rclbf_high_goal_time)
    t_index_mpc1 = min(t_index, mpc1_high_goal_time)
    t_index_mpc25 = min(t_index, mpc25_high_goal_time)

    quad_clbf.set_xy([x_rclbf_high[t_index_clbf, 0], x_rclbf_high[t_index_clbf, 1]])
    quad_clbf._angle = -np.rad2deg(x_rclbf_high[t_index_clbf, 2])
    quad_mpc1.set_xy([x_mpc1_high[t_index_mpc1, 0], x_mpc1_high[t_index_mpc1, 1]])
    quad_mpc1._angle = -np.rad2deg(x_mpc1_high[t_index_mpc1, 2])
    quad_mpc25.set_xy([x_mpc25_high[t_index_mpc25, 0], x_mpc25_high[t_index_mpc25, 1]])
    quad_mpc25._angle = -np.rad2deg(x_mpc25_high[t_index_mpc25, 2])
    return quad_clbf, quad_mpc1, quad_mpc25


anim = FuncAnimation(fig, animate_high, interval=1000 / 30, frames=int(7.5 * 30))
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
# anim.save('/home/cbd/Downloads/pvtol_obs_high.mov')
plt.show()
plt.draw()