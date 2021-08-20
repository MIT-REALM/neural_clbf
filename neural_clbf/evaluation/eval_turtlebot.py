#!/usr/bin/python

from neural_clbf.controllers import NeuralCLBFController
from integration.integration.turtlebot_scripts import run_turtlebot_node
import torch

@torch.no_grad()
def eval_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "~/neural_clbf/saved_models/aas/turtlebot/"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_dir + "version_0.ckpt")

    # setup ros nodes and run controller
    run_turtlebot_node.run_turtlebot(neural_controller)


if __name__ == "__main__":
    eval_turtlebot()
