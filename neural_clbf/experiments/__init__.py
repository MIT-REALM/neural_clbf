from warnings import warn

from .experiment import Experiment
from .experiment_suite import ExperimentSuite

from .clf_contour_experiment import CLFContourExperiment
from .clf_verification_experiment import CLFVerificationExperiment
from .bf_contour_experiment import BFContourExperiment
from .lf_contour_experiment import LFContourExperiment
from .rollout_time_series_experiment import RolloutTimeSeriesExperiment
from .rollout_norm_experiment import RolloutNormExperiment
from .rollout_state_space_experiment import RolloutStateSpaceExperiment
from .rollout_success_rate_experiment import RolloutSuccessRateExperiment
from .car_s_curve_experiment import CarSCurveExperiment
from .obs_bf_verification_experiment import ObsBFVerificationExperiment


__all__ = [
    "Experiment",
    "ExperimentSuite",
    "CLFContourExperiment",
    "CLFVerificationExperiment",
    "BFContourExperiment",
    "LFContourExperiment",
    "RolloutTimeSeriesExperiment",
    "RolloutStateSpaceExperiment",
    "RolloutSuccessRateExperiment",
    "RolloutNormExperiment",
    "CarSCurveExperiment",
    "ObsBFVerificationExperiment",
]

try:
    from .turtlebot_hw_state_feedback_experiment import (  # noqa: F401
        TurtlebotHWStateFeedbackExperiment,
    )
    from .turtlebot_hw_obs_feedback_experiment import (  # noqa: F401
        TurtlebotHWObsFeedbackExperiment,
    )

    __all__.append("TurtlebotHWStateFeedbackExperiment")
    __all__.append("TurtlebotHWObsFeedbackExperiment")
except ImportError:
    warn("Could not import HW module; is ROS installed?")
