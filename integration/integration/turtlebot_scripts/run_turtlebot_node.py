#!/usr/bin/env python3

"""
Script combining functionality of all publisher and subscriber
nodes for the turtlebot into one interface
"""
import rospy
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import BatteryState, LaserScan
import tf

from typing import Union

import torch

from .battery_status import battery
from .laser_data import LidarMonitor
from neural_clbf.experiments import (
    ExperimentSuite,
    TurtlebotHWStateFeedbackExperiment,
    TurtlebotHWObsFeedbackExperiment,
)


class TurtleBotNode(object):
    def __init__(self):
        """

        Initializes publishers, subscribers, and various
        other necessary variables.

        """

        # create a node name run_turtlebot_node
        rospy.init_node("run_turtlebot_node", anonymous=True)

        # set update rate; i.e. how often we send commands (Hz)
        self.rate = rospy.Rate(10)

        # create transform listener to transform coords from turtlebot frame to
        # global frame
        self.listener = tf.TransformListener()

        # create a position of type Point
        self.position = Point()

        # create an object to send commands to turtlebot of type Twist
        # allows us to send velocity and rotation commands
        self.move_command = Twist()

        # set odom_frame to the /odom topic being published by the turtlebot
        self.odom_frame = "/odom"

        # create a publisher node to send velocity commands to turtlebot
        self.command_publisher = rospy.Publisher("cmd_vel", Twist, queue_size=10)

        # create a subscriber to get measurements from lidar sensor.
        self.lidar_monitor = LidarMonitor()
        rospy.Subscriber("/scan", LaserScan, self.lidar_monitor.scan_callback)

        # create a subscriber for battery level
        rospy.Subscriber("battery_state", BatteryState, battery)

        rospy.on_shutdown(self.shutdown)

        # find the coordinate conversion from the turtlebot to the ground truth frame
        try:
            self.listener.waitForTransform(
                self.odom_frame, "base_footprint", rospy.Time(), rospy.Duration(1.0)
            )
            self.base_frame = "base_footprint"
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            try:
                self.listener.waitForTransform(
                    self.odom_frame, "base_link", rospy.Time(), rospy.Duration(1.0)
                )
                self.base_frame = "base_link"
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo(
                    "Cannot find transform between odom and base_link or base_footprint"
                )
                rospy.signal_shutdown("tf Exception")

        # bool flag to stop commands from running multiple times
        # (see run_turtlebot() below)
        self.first_turn = True

    def shutdown(self):
        """

        Publishes empty twist command to turtlebot to make it stop moving

        """

        self.command_publisher.publish(Twist())
        rospy.sleep(1)


def run_turtlebot(controller, log_dir: str, obs_feedback: bool = False):
    """

    Creates an experiment and turtlebot object
    and runs the experiment file

    args:
        controller: a neural_clbf.controllers.Controller subclass
        log_dir: the directory to save the results
        obs_feedback: if True, use an observation feedback experiment
    """
    # Initialize an instance of the controller
    start_tensor = torch.tensor([[-3.0, -3.0, 0]])

    turtle = TurtleBotNode()
    experiment: Union[
        TurtlebotHWObsFeedbackExperiment, TurtlebotHWStateFeedbackExperiment
    ]
    if obs_feedback:
        experiment = TurtlebotHWObsFeedbackExperiment(
            "turtlebot_hw_experiment",
            turtle.command_publisher,
            turtle.rate,
            turtle.listener,
            turtle.move_command,
            turtle.odom_frame,
            turtle.base_frame,
            turtle.lidar_monitor,
            start_x=start_tensor,
        )
    else:
        # Default to state-feedback
        experiment = TurtlebotHWStateFeedbackExperiment(
            "turtlebot_hw_experiment",
            turtle.command_publisher,
            turtle.rate,
            turtle.listener,
            turtle.move_command,
            turtle.odom_frame,
            turtle.base_frame,
            start_x=start_tensor,
        )
    experiment_suite = ExperimentSuite([experiment])

    while not rospy.is_shutdown():

        # if statement to prevent same command from
        # running indefinitely
        if turtle.first_turn is True:
            experiment_suite.run_all_and_save_to_csv(controller, log_dir)

        turtle.first_turn = False
