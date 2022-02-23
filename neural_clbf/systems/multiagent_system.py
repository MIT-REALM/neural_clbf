"""Define a dynamical system for turtlebot and quadrotor multiagent system"""
from msilib.schema import Control
from typing import Tuple, List, Optional

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from .utils import grav, Scenario


class Multiagent(ControlAffineSystem):
    """
    Represents a quadrotor and turtlebot working in tandem.

    The system has states

        x_t = [x y theta]
        x_quad = [x y z vx vy vz]
        Q = [x y theta x y z]

    representing the postiion, orientation, and velocities of the quadrotor

    and it has control inputs 

        u = [] #TODO @bethlow

    """


    #Number of states and controls #TODO 
    N_DIMS = 0
    N_CONTROLS = 0


    def __init__(
            self,
            nominal_params: Scenario,
            dt: float = 0.01,
            controller_dt: Optional[float] = None,
        ):
            """
            Initialize the system.

            args:
                nominal_params: a dictionary giving the parameter values for the system.
                                Requires keys ["m"]
                dt: the timestep to use for the simulation
                controller_dt: the timestep for the LQR discretization. Defaults to dt
            raises:
                ValueError if nominal_params are not valid for this system
            """
            super().__init__(nominal_params, dt, controller_dt)