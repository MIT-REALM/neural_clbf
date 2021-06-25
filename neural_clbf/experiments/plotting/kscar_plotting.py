import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
import numpy as np
import seaborn as sns
import neural_clbf.setup.commonroad as commonroad_loader  # type: ignore
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2  # type: ignore


# make sure that the import worked
assert commonroad_loader


def main():
    # Beautify plots
    sns.set_theme(context="talk", style="white")
    mpc_color = sns.color_palette("pastel")[0]
    mpc_ecolor = sns.color_palette("pastel")[0] + (1.0,)
    rclbf_color = sns.color_palette("pastel")[1]

    # Load the data from the CSVs
    filename = "sim_traces/kscar_rCLBF-QP_dt=0-01_omega_ref=1-5-sine.csv"
    x_rclbf = np.loadtxt(filename, delimiter=",", skiprows=1)
    filename = "sim_traces/kscar_rmpc_dt=0-1_omega_ref=1-5-sine.csv"
    x_mpc1 = np.loadtxt(filename, delimiter=",", skiprows=1)
    filename = "sim_traces/kscar_rmpc_dt=0-25_omega_ref=1-5-sine.csv"
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

    print("KSCar maximum tracking error:")
    print(f"\trCLBF-QP: {xy_err_rclbf.max()}")
    print(f"\trMPC dt0.1: {xy_err_mpc1.max()}")
    print(f"\trMPC dt0.25: {xy_err_mpc25.max()}")

    # Animate!

    class KSCarPatches(object):
        """Matplotlib patches representing a KSCar"""

        def __init__(self, ecolor, fcolor, hatch=None):
            super(KSCarPatches, self).__init__()

            # get the dimensions
            self.lf = parameters_vehicle2().a
            self.lr = parameters_vehicle2().b

            self.wheel_width = parameters_vehicle2().w
            self.wheel_length = 0.15 * parameters_vehicle2().l

            # Define a patch for the front and rear wheels
            self.front_wheel = patches.Rectangle(
                (0.0, 0.0),
                self.wheel_width,
                self.wheel_length,
                linewidth=1,
                facecolor=fcolor,
                edgecolor=ecolor,
                hatch=hatch,
            )
            self.rear_wheel = patches.Rectangle(
                (0.0, 0.0),
                self.wheel_width,
                self.wheel_length,
                linewidth=1,
                facecolor=fcolor,
                edgecolor=ecolor,
                hatch=hatch,
            )

            # And a patch for the center of mass
            self.com = patches.Circle((0.0, 0.0), self.wheel_width / 10, color="k")

            # And a patch connecting them
            self.connector_path = Path([[0.0, 0.0], [self.lr + self.lf, 0.0]])
            self.connector = patches.PathPatch(self.connector_path, color="k", lw=4)

        def get_front_wheel_xy(self, x, y, psi):
            xf = x + self.lf * np.cos(psi)
            yf = y + self.lf * np.sin(psi)
            return xf, yf

        def get_front_wheel_rotation_rad(self, psi, delta):
            return psi + delta

        def get_rear_wheel_rotation_rad(self, psi):
            return psi

        def get_rear_wheel_xy(self, x, y, psi):
            xr = x - self.lr * np.cos(psi)
            yr = y - self.lr * np.sin(psi)
            return xr, yr

        def update_with_state(self, x, y, psi, delta, ax):
            front_wheel_angle = self.get_front_wheel_rotation_rad(psi, delta)
            front_xform = mpl.transforms.Affine2D()
            front_xform.translate(-self.wheel_width / 2.0, -self.wheel_length / 2.0)
            front_xform.rotate(front_wheel_angle)
            front_xform.translate(*self.get_front_wheel_xy(x, y, psi))
            front_xform += ax.transData
            self.front_wheel.set_transform(front_xform)

            rear_wheel_angle = self.get_rear_wheel_rotation_rad(psi)
            rear_xform = mpl.transforms.Affine2D()
            rear_xform.translate(-self.wheel_width / 2.0, -self.wheel_length / 2.0)
            rear_xform.rotate(rear_wheel_angle)
            rear_xform.translate(*self.get_rear_wheel_xy(x, y, psi))
            rear_xform += ax.transData
            self.rear_wheel.set_transform(rear_xform)

            com_xform = mpl.transforms.Affine2D()
            com_xform = com_xform.translate(x, y)
            com_xform += ax.transData
            self.com.set_transform(com_xform)

            connector_xform = mpl.transforms.Affine2D()
            connector_xform.rotate(rear_wheel_angle)
            connector_xform.translate(*self.get_rear_wheel_xy(x, y, psi))
            connector_xform += ax.transData
            self.connector.set_transform(connector_xform)

        def patches(self):
            return self.front_wheel, self.rear_wheel, self.com, self.connector

    # Animate CLBF
    fig, ax = plt.subplots(figsize=(25, 15))

    # Plot reference
    x_ref = x_rclbf[:, -2]
    y_ref = x_rclbf[:, -1]
    ax.plot(
        x_ref,
        y_ref,
        linestyle="dotted",
        label="Ref",
        color="black",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend()
    ax.set_ylim([np.min(y_ref) - 3, np.max(y_ref) + 3])
    ax.set_xlim([np.min(x_ref) - 3, np.max(x_ref) + 3])

    # Make the path
    (rclbf_path,) = ax.plot(
        xy_rclbf[:, 0] + x_ref[:],
        xy_rclbf[:, 1] + y_ref[:],
        lw=2,
        color=rclbf_color,
        label="rCLBF",
    )
    (mpc1_path,) = ax.plot(
        xy_mpc1[:, 0] + x_ref[:],
        xy_mpc1[:, 1] + y_ref[:],
        lw=2,
        color=mpc_color,
        linestyle="dashed",
        label="rMPC ($dt=0.1$)",
    )
    (car_path,) = ax.plot(
        [], [], lw=2, color=mpc_color, linestyle="solid", label="rMPC ($dt=0.25$)"
    )

    # Make the car icon
    kscar_patches = KSCarPatches(mpc_ecolor, mpc_ecolor)

    car_patches = kscar_patches.patches()
    for car_patch in car_patches:
        ax.add_patch(car_patch)

    ax.legend(fontsize=25, loc="upper left")

    def animate_clbf(i):
        # i is the frame. At 30 fps, t = i/30
        delta_t = t_sim / num_timesteps
        t_index = int((i / 30) / delta_t)

        x = xy_mpc25[t_index, 0] + x_ref[t_index]
        y = xy_mpc25[t_index, 1] + y_ref[t_index]
        psi = x_mpc25[t_index, 4] + x_mpc25[t_index, 6]
        delta = x_mpc25[t_index, 2]

        kscar_patches.update_with_state(x, y, psi, delta, ax)

        car_path.set_data(
            xy_mpc25[:t_index, 0] + x_ref[:t_index],
            xy_mpc25[:t_index, 1] + y_ref[:t_index],
        )

        return kscar_patches.patches()

    anim = FuncAnimation(fig, animate_clbf, interval=1000 / 30, frames=int(5 * 30))
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    anim.save("/home/cbd/Downloads/kscar_mpc25.mov")
    plt.show()
    plt.draw()


if __name__ == "__main__":
    main()
