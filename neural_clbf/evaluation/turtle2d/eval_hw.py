#!/usr/bin/python

from neural_clbf.controllers import NeuralObsBFController
from integration.integration.turtlebot_scripts import run_turtlebot_node
import torch


@torch.no_grad()
def eval_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/perception/turtlebot2d/commit_8439378/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(
        log_dir + "v0_ep72.ckpt"
    )

    # setup ros nodes and run controller
    run_turtlebot_node.run_turtlebot(neural_controller, log_dir, obs_feedback=True)


if __name__ == "__main__":
    eval_turtlebot()
