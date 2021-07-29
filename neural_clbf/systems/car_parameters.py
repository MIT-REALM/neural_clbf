"""Includes parameters taken from the CommonRoad benchmark models for car dynamics"""


class VehicleParameters(object):
    """A restricted set of the commonroad vehicle parameters, which can be
    found as forked at https://github.com/dawsonc/commonroad-vehicle-models"""

    def __init__(self):
        super(VehicleParameters, self).__init__()

        self.steering_min = -1.066  # minimum steering angle [rad]
        self.steering_max = 1.066  # maximum steering angle [rad]
        self.steering_v_min = -0.4  # minimum steering velocity [rad/s]
        self.steering_v_max = 0.4  # maximum steering velocity [rad/s]

        self.longitudinal_a_max = 11.5  # maximum absolute acceleration [m/s^2]

        self.tire_p_dy1 = 1.0489  # Lateral friction Muy
        self.tire_p_ky1 = -21.92  # Maximum value of stiffness Kfy/Fznom

        # distance from spring mass center of gravity to front axle [m]  LENA
        self.a = 0.3048 * 3.793293
        # distance from spring mass center of gravity to rear axle [m]  LENB
        self.b = 0.3048 * 4.667707
        self.h_s = 0.3048 * 2.01355  # M_s center of gravity above ground [m]  HS
        self.m = 4.4482216152605 / 0.3048 * (74.91452)  # vehicle mass [kg]  MASS
        # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
        self.I_z = 4.4482216152605 * 0.3048 * (1321.416)
