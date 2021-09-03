#!/usr/bin/env python3

# have to define variable here to prevent error with ros
batteryLevel = 100
import integration.integration.turtlebot_scripts.message_creator as message_creator  # noqa


def battery(data):
    """

    Gets battery percentage from subscribed ROS topic and
    converts it into a percentage out of 100.

    """
    global batteryLevel
    message_creator.batteryLevel = batteryLevel

    # this function was found empirically through multiple tests
    # with a turtlebot. It is an estimate.
    batteryLevel = 100 * 1.74 * (data.percentage - 0.986) / (1.225 - 0.986)
