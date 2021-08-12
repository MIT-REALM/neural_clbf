#!/usr/bin/env

import rospy


def create_message(move_command, batteryLevel, distance, angle):
    """
    
    Generates a message containing information on the turtlebot's
    current position, rotation, command, and battery level and then
    outputs this information to the console.
    
    """
    
    if(batteryLevel >= 10):
        msg = """
        Sending commands to turtlebot...
        --------------------------------
        Linear velocity: %f
        Angular Velocity (CCW): %f
        --------------------------------
        Odometry

        Distance travelled (meters): %f

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

        Distance travelled (meters): %f

        Angle (degrees): %f
        --------------------------------
        Estimated battery level: %f/100
        BATTERY LEVEL LOW. CHARGE SOON.
        --------------------------------
        
        --------------------------------

        """

    # write the above message to the console with velocity, position, battery level values
    rospy.loginfo(msg, round(move_command.linear.x, 2), round(move_command.angular.z, 2), distance.x, angle, batteryLevel)
    

    
