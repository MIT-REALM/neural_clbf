#!/usr/bin/env

from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point
import rospy
import tf


def get_odom(listener, odom_frame, base_frame):
    """
    
    Finds coordinate transform between odometry and ground truth frames and returns the
    position and z rotation of the turtlebot

    """

    try:
        (trans,rot) = listener.lookupTransform(odom_frame, base_frame, rospy.Time(0))
        rotation = euler_from_quaternion(rot)
        
    except(tf.Exception, tf.ConnectivityException, tf.LookupException):
        rospy.loginfo("tf Exception")
        return
    
    return (Point(*trans), rotation[2])

