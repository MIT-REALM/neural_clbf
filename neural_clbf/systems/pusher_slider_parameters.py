"""Defines some standard values for the Pusher Slider systems"""


class PusherSliderParameters(object):
    """
    PusherSliderParameters
    Description:
        A set of parameters which correspond to pusher slider models as found in [1].
    References:
        [1] = Hogan, François Robert, and Alberto Rodriguez.
              "Feedback control of the pusher-slider system: A story of hybrid and underactuated contact dynamics."
              arXiv preprint arXiv:1611.08268 (2016).
    """

    def __init__(self):
        super(PusherSliderParameters, self).__init__()

        self.s_width = 0.09  # m
        self.s_length = 0.09  # m
        self.s_mass = 1.05  # kg
        self.ps_cof = 0.3  # Pusher-to-Slider Coefficient of Friction
        self.st_cof = 0.35  # Slider-to-Table Coefficient of Friction

        self.p_radius = 0.01  # Radius of the Pusher representation (circle)

        self.p_x = self.s_width / 2.0