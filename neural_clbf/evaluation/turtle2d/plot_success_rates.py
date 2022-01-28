"""Plot data gathered for success and collision rates"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_success_rate():
    # Define data, gathered from various scripts, in tidy data format
    data = []

    # Neural oCBF/oCLF data generated using eval_turtlebot_neural_cbf_mpc_success_rates
    data += [
        {
            "Algorithm": "Observation-based CBF/CLF (ours)",
            "Metric": "Goal-reaching rate",
            "Value": 0.932,
        },
        {
            "Algorithm": "Observation-based CBF/CLF (ours)",
            "Metric": "Safety rate",
            "Value": 1.0,
        },
        {
            "Algorithm": "Observation-based CBF/CLF (ours)",
            "Metric": "Avg. time to goal (s)",
            "Value": 2.1838412017167395,
        },
    ]

    # State-based CBF data also generated using
    # eval_turtlebot_neural_cbf_mpc_success_rates
    data += [
        {
            "Algorithm": "State-based CBF/CLF",
            "Metric": "Goal-reaching rate",
            "Value": 0.546,
        },
        {"Algorithm": "State-based CBF/CLF", "Metric": "Safety rate", "Value": 0.626},
        {
            "Algorithm": "State-based CBF/CLF",
            "Metric": "Avg. time to goal (s)",
            "Value": 1.9382783882783883,
        },
    ]

    # MPC data also generated using eval_turtlebot_neural_cbf_mpc_success_rates
    data += [
        {
            "Algorithm": "MPC",
            "Metric": "Goal-reaching rate",
            "Value": 0.904,
        },
        {"Algorithm": "MPC", "Metric": "Safety rate", "Value": 0.996},
        {
            "Algorithm": "MPC",
            "Metric": "Avg. time to goal (s)",
            "Value": 2.093,
        },
    ]

    # PPO data gathered by running
    # python scripts/test_policy.py \
    #   data/2021-08-13_ppo_turtle2d/2021-08-13_15-23-36-ppo_turtle2d_s0 \
    #   --len 100 --episodes 100 --norender
    # in the safety_starter_agents directory, with the turtle2d env.
    # Steps are converted to time with timestep 0.1
    data += [
        {"Algorithm": "PPO", "Metric": "Goal-reaching rate", "Value": 256 / 500},
        {"Algorithm": "PPO", "Metric": "Safety rate", "Value": 1 - 57 / 500},
        {"Algorithm": "PPO", "Metric": "Avg. time to goal (s)", "Value": 0.1 * 38.07},
    ]

    # Convert to dataframe
    df = pd.DataFrame(data)

    # Convert rates to percentages
    rate_mask = (df["Metric"] == "Goal-reaching rate") | (df["Metric"] == "Safety rate")
    df.loc[rate_mask, "Value"] *= 100

    # Plot!
    sns.set_theme(context="talk", style="white")
    sns.set_style({"font.family": "serif"})
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})
    left_plot_mask = df["Metric"] != "Avg. time to goal (s)"
    sns.barplot(
        x="Metric", y="Value", hue="Algorithm", ax=axs[0], data=df[left_plot_mask]
    )
    axs[0].plot(axs[0].get_xlim(), [100, 100], "k--")
    axs[0].set_ylabel("%")
    axs[0].set_xlabel("")
    axs[0].legend(loc="lower left")
    right_plot_mask = df["Metric"] == "Avg. time to goal (s)"
    sns.barplot(
        x="Metric", y="Value", hue="Algorithm", ax=axs[1], data=df[right_plot_mask]
    )
    axs[1].set_ylabel("")
    axs[1].set_xlabel("")
    axs[1].get_legend().remove()

    plt.show()


if __name__ == "__main__":
    plot_success_rate()
