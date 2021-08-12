#!/usr/bin/env python3

"""
Script combining functionality of all publisher and subscriber
nodes for the turtlebot into one interface
"""

# import python and ros libraries
import rospy
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import BatteryState, LaserScan
from actionlib_msgs.msg import *
from sound_play.libsoundplay import SoundClient

# import other realm-specific scripts
import battery_status
import laser_data
from turtlebot_functions import *
from neural_clbf.experiments.turtlebot_time_series_experiment import RealTimeSeriesExperiment
import torch


# just setting an initial value for these; otherwise an error gets thrown
# might be able to just toss these later? depends on how script changes
posX = 0
posY = 0
batteryLevel = 100


class TurtleBot(object):
    
    def __init__(self):
        """
        
        Initializes publishers, subscribers, and various
        other necessary variables.

        IMPORTANT: When replacing the control script, you must
        modify the "self.command = ..." line (line 102) within this function.
        This line of code passes several important arguments to
        the control script that are needed to call various functions.
        
        """
        # create a node name run_turtlebot_node
        rospy.init_node('run_turtlebot_node', anonymous=True)

        # might try getting sound functionality but hasn't worked yet
        # sound_client = SoundClient()
        # sound = rospy.Publisher('sound', SoundClient, queue_size=10)
        # rospy.sleep(2)
        # sound.publish('/home/realm/Downloads/chime_up.wav')

        # set update rate; i.e. how often we send commands (Hz)
        self.rate = rospy.Rate(10)

        # create transform listener to transform coords from turtlebot frame to absolute frame
        self.listener = tf.TransformListener()

        # create a position of type Point
        # TODO can probably delete this since it gets set and updated within other scripts
        self.position = Point()
        
        # create an object to send commands to turtlebot of type Twist
        # allows us to send velocity and rotation commands
        self.move_command = Twist()

        # set odom_frame to the /odom topic being published by the turtlebot
        self.odom_frame = '/odom'

        # create a publisher node to send velocity commands to turtlebot
        self.command_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        # set the starting point [x, y, theta]
        self.initial_position = [1, 1, 0]

        # create a subscriber to get measurements from lidar sensor.
        # currently not used, but is left here in case lidar measurements
        # are needed in the future. See also the laser_data.py script.
        rospy.Subscriber('/scan', LaserScan, laser_data.get_laser_data)

        # create a subscriber for battery level
        rospy.Subscriber('battery_state', BatteryState, battery_status.battery)

        #TODO what's this line do?
        rospy.on_shutdown(self.shutdown)


        # find the coordinate conversion from the turtlebot to the ground truth frame
        try:
            self.listener.waitForTransform(self.odom_frame, "base_footprint", rospy.Time(), rospy.Duration(1.0))
            self.base_frame = "base_footprint"
        except(tf.Exception, tf.LookupException, tf.ConnectivityException):
            try:
                self.listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except(tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")

        # IMPORTANT: When substituting in a new control script, you must update this line
        self.command = turtlebot_command(self.command_publisher, self.rate, self.listener, self.move_command, self.odom_frame, self.base_frame)

        # bool flag to stop commands from running multiple times (see run_turtlebot() below)
        self.first_turn = True
    

    def shutdown(self):
        """
        
        Publishes empty twist command to turtlebot to make it stop moving

        """

        self.command_publisher.publish(Twist())
        rospy.sleep(1)
    

# function that sends velocity commands to turtlebot, calls other functions. Run by main function (see below)
def run_turtlebot():
    """
    
    Sends commands to the turtlebot

    """
    # Initialize an instance of the controller
    # TODO: do we need an argument here?
    turtle = TurtleBot()
    controller = RealTimeSeriesExperiment(turtle.command_publisher, turtle.rate, turtle.listener, turtle.position, turtle.move_command, turtle.odom_frame, turtle.base_frame, "initial test", torch.Tensor(turtle.initial_position))

    while not rospy.is_shutdown():

        # if statement to prevent same command from
        # running indefinitely
        if turtle.first_turn is True:
            print("")
            # TODO call the controller here
            # do we need a specific function?

        turtle.first_turn = False
        

# main function; executes the run_turtlebot function until we hit control + C
if __name__ == '__main__':
    try:
        run_turtlebot()
    except rospy.ROSInterruptException:
        pass
