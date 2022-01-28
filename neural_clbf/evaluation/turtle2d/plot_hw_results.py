import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import torch


@torch.no_grad()
def plot_animated_traces():
    # Load the trace from the experiment
    run = "dynamic_5"
    log_dir = "saved_models/perception/turtlebot2d/commit_8439378/hw/"
    if run == "static_6":
        log_dir += "2021-09-07_11_31_09/"
    elif run == "dynamic_5":
        log_dir += "2021-09-07_11_46_47/"

    results_df = pd.read_csv(log_dir + "turtlebot_hw_experiment.csv")
    t = 1 / 20.0 * results_df.index

    # Set up the two plots
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(6.5, 10)
    h_ax, V_ax = axs

    # Set up h plot
    (h_line,) = h_ax.plot([], [])
    h_ax.set_xlim([0, 35])
    h_ax.set_ylim([-2, 0.2])
    h_ax.plot([0, 35], [0, 0], "k:")
    h_ax.set_xlabel("$t$")
    h_ax.set_ylabel("$h$")

    # Set up V plot
    (V_line,) = V_ax.plot([], [])
    V_ax.set_xlim([0, 35])
    V_ax.set_ylim([0.0, 20.0])
    # V_ax.plot(t, 0 * t, "k:")
    V_ax.set_xlabel("$t$")
    V_ax.set_ylabel("$V$")

    def animate(i):
        # Get time at frame i
        h_line.set_data(t[:i], results_df["h"][:i])
        V_line.set_data(t[:i], results_df["V"][:i])

        return h_line, V_line

    ani = animation.FuncAnimation(fig, animate, 3500, interval=1000.0 / 20.0, blit=True)
    # plt.show()
    writervideo = animation.FFMpegWriter(fps=20)
    ani.save(run + "_h_V.mp4", writer=writervideo)
    plt.close()


if __name__ == "__main__":
    plot_animated_traces()
