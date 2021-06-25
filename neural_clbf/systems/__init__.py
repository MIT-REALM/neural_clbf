from warnings import warn

from .control_affine_system import ControlAffineSystem
from .quad2d import Quad2D
from .quad3d import Quad3D
from .neural_lander import NeuralLander
from .inverted_pendulum import InvertedPendulum
from .kinematic_single_track_car import KSCar
from .single_track_car import STCar
from .segway import Segway


__all__ = [
    "ControlAffineSystem",
    "InvertedPendulum",
    "Quad2D",
    "Quad3D",
    "NeuralLander",
    "KSCar",
    "STCar",
    "Segway",
]

try:
    from .f16 import F16  # noqa

    __all__.append("F16")
except ImportError:
    warn("Could not import F16 module; is AeroBench installed")
