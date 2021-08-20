#!/usr/bin/env python3

"""
Script combining functionality of all publisher and subscriber
nodes for the turtlebot into one interface
"""

# import python and ros libraries
from neural_clbf.controllers.neural_cbf_controller import NeuralCBFController
from neural_clbf.experiments import experiment
import rospy
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import BatteryState, LaserScan
from actionlib_msgs.msg import *
# from sound_play.libsoundplay import SoundClient

# import other realm-specific scripts
from .battery_status import battery, batteryLevel
from .laser_data import get_laser_data
# from turtlebot_functions import *
from neural_clbf.experiments.real_time_series_experiment import RealTimeSeriesExperiment
import torch
import tf
from neural_clbf.systems import TurtleBot
from neural_clbf.controllers import NeuralCLBFController
from numpy import pi


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

        # create a subscriber to get measurements from lidar sensor.
        # currently not used, but is left here in case lidar measurements
        # are needed in the future. See also the laser_data.py script.
        rospy.Subscriber('/scan', LaserScan, get_laser_data)

        # create a subscriber for battery level
        rospy.Subscriber('battery_state', BatteryState, battery)

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
        # self.command = turtlebot_command(self.command_publisher, self.rate, self.listener, self.move_command, self.odom_frame, self.base_frame)

        # bool flag to stop commands from running multiple times (see run_turtlebot() below)
        self.first_turn = True
    

    def shutdown(self):
        """
        
        Publishes empty twist command to turtlebot to make it stop moving

        """

        self.command_publisher.publish(Twist())
        rospy.sleep(1)
    

# 
def run_turtlebot(neural_controller):
    """
    
    Creates an experiment and turtlebot object
    and runs the experiment file

    """
    # Initialize an instance of the controller
    start_tensor  = torch.tensor(
        [[-1, -1, 0.0]]
    )
    list1 = [0,1]
    list2 = [0,1]
    list3 = ["test", "test"]
    list4 = ["test", "test"]
    turtle = TurtleBot()
    experiment = RealTimeSeriesExperiment(turtle.command_publisher, turtle.rate, turtle.listener, turtle.move_command, 
    turtle.odom_frame, turtle.base_frame, "real experiment", start_x=start_tensor, plot_x_indices=list1, plot_x_labels=list3, 
    plot_u_indices=list2, plot_u_labels=list4)
    neural_controller.experiment_suite.experiments = [experiment]


    while not rospy.is_shutdown():

        # if statement to prevent same command from
        # running indefinitely
        if turtle.first_turn is True:
            experiment.run(neural_controller)

        turtle.first_turn = False
        
def run():
    # main function; executes the run_turtlebot function until we hit control + C
    if __name__ == '__main__':
        try:
            run_turtlebot()
        except rospy.ROSInterruptException:
            pass
