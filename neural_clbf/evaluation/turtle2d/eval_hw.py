#!/usr/bin/python

from neural_clbf.controllers import NeuralObsBFController
from integration.integration.turtlebot_scripts import run_turtlebot_node
import torch

"""
Notes:
Ran 5 runs in a static environment. Some missed the goal tolerance (detection off
of the server), but can be cut to the goal-reaching time for a good video. Good
sample of different behaviors.

3 static bug-trap only tests

dynamic obstacle tests:
1 failed since I moved the obstacles too much (made it impossible to find 0.9 V_hit)

"""


@torch.no_grad()
def eval_turtlebot():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/perception/turtlebot2d/commit_8439378/"
    neural_controller = NeuralObsBFController.load_from_checkpoint(
        log_dir + "v0_ep72.ckpt"
    )

    # neural_controller.debug_mode_goal_seeking = True
    # neural_controller.debug_mode_exploratory = True

    # setup ros nodes and run controller
    run_turtlebot_node.run_turtlebot(
        neural_controller, log_dir + "hw/", obs_feedback=True
    )


if __name__ == "__main__":
    eval_turtlebot()
