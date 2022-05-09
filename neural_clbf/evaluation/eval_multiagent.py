""""""
import pandas as pd
import torch

from neural_clbf.controllers.multiagent_controller import MultiagentController
from neural_clbf.experiments import RolloutMultiagentStateSpaceExperiment

# def eval_multiagent():
#     # Load the checkpoint file. This should include the experiment suite used during
#     # training.
#     log_dir = "saved_models/multiagent/commit_63ca7c9/"
#     neural_controller = NeuralCLBFController.load_from_checkpoint(
#         log_dir + "version_0.ckpt"
#     )
#     # Run the experiments and save the results
#     neural_controller.experiment_suite.run_all_and_save_to_csv(
#         neural_controller, log_dir + "experiments"
#     )

def plot_multiagent():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    # log_dir = "saved_models/multiagent/commit_63ca7c9/"
    # neural_controller = NeuralCLBFController.load_from_checkpoint(
    #     log_dir + "version_0.ckpt"
    # 
    neural_controller = MultiagentController()
    #Turtlebot array of zeros
    #Crazyflie 
    experiment = RolloutMultiagentStateSpaceExperiment("Crazyflie", torch.tensor([[1.0,1,1,1,1,1,0,1,0,0,0,0]]), 3, "x", 4, "y", 5, "z")
    # Set the path to load from

    experiment.run_and_plot(neural_controller, display_plots=True)
    # experiment_dir = "experiments/2021-03-13_11_11_33/"

    # Rollout State Space Experiment
    # results_df = pd.read_csv(experiment_dir + "/Multiagent Experiment.csv")
    # neural_controller.experiment_suite.experiments[1].plot(
    #     neural_controller, results_df, display_plots=True
    # )


if __name__ == "__main__":
    # eval_multiagent()
    plot_multiagent()
