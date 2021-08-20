#! /usr/bin/env python
distance_to_object = 0.0
distance_to_object_right = 0.0
distance_to_object_left = 0.0

def get_laser_data(msg):
    """
    
    Outputs readings from the turtlebot's lidar sensor.
    Currently not in use, but may come in handy if obstacle 
    detection is needed at a later date. Only returns readings
    in front of the turtlebot (0 degrees), to the left of the
    turtlebot (90 degrees), and to the right of the turtlebot
    (270 degrees). Naturally, measurements from all around the
    turtlebot can be obtained with a small modification to this
    function by iterating over all angles about the turtlebot.
    
    """
    global distance_to_object
    global distance_to_object_left
    global distance_to_object_right
    distance_to_object = msg.ranges[0]
    distance_to_object_left = msg.ranges[90]
    distance_to_object_right = msg.ranges[270]

    return distance_to_object, distance_to_object_left, distance_to_object_right
