import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_turtle_training_curves():
    # Load training and validation data
    log_dir = "saved_models/perception/turtlebot2d/commit_8439378/"
    train_log_file = "run-commit_8439378_version_0_total_loss_train.csv"
    val_log_file = "run-commit_8439378_version_0_total_loss_val.csv"
    train_loss_df = pd.read_csv(log_dir + train_log_file)
    train_loss_df["Dataset"] = "Training"
    val_loss_df = pd.read_csv(log_dir + val_log_file)
    val_loss_df["Dataset"] = "Validation"

    # Concatenate the datasets
    loss_df = pd.concat([train_loss_df, val_loss_df])
    loss_df.rename(columns={"Value": "Total Loss"}, inplace=True)

    # We took the model from epoch 72, which occured approx at step 19200, so
    # crop the data to that point
    data_mask = loss_df["Step"] < 19200

    # Plot!
    sns.set_theme(context="paper", style="white")
    sns.set_style({"font.family": "serif"})
    ax = sns.lineplot(
        x="Step",
        y="Total Loss",
        hue="Dataset",
        data=loss_df[data_mask],
        linewidth=2,
    )
    ax.set(yscale="log")
    fig = plt.gcf()
    fig.set_size_inches(4, 2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_turtle_training_curves()
