#!/usr/bin/env python3

import rospy

# dummy varabv
batteryLevel = 100


def create_message(move_command, position, angle):
    """

    Generates a message containing information on the turtlebot's
    current position, rotation, command, and battery level and then
    outputs this information to the console.

    """

    if batteryLevel >= 10:
        msg = """
        Sending commands to turtlebot...
        --------------------------------
        Linear velocity: %f
        Angular Velocity (CCW): %f
        --------------------------------
        Odometry

        X Position (meters): %f

        Y Position (meters): %f

        Angle (degrees): %f
        --------------------------------
        Estimated battery level: %f/100
        --------------------------------

        --------------------------------

        """

    else:
        msg = """
        Sending commands to turtlebot...
        --------------------------------
        Linear velocity: %f
        Angular Velocity (CCW): %f
        --------------------------------
        Odometry

        X Position (meters): %f

        Y Position (meters): %f

        Angle (degrees): %f
        --------------------------------
        Estimated battery level: %f/100
        BATTERY LEVEL LOW. CHARGE SOON.
        --------------------------------

        --------------------------------

        """

    # write the above message to the console with velocity, position,
    # and battery level values
    rospy.loginfo(
        msg,
        round(move_command.linear.x, 2),
        round(move_command.angular.z, 2),
        position.x,
        position.y,
        angle,
        batteryLevel,
    )
