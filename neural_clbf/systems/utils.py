"""Defines useful constants and helper functions for dynamical systems"""
from typing import Dict, List

import numpy as np
import scipy.linalg


# Gravitation acceleration
grav = 9.80665

# Define a type alias for parameter scenarios
Scenario = Dict[str, float]
ScenarioList = List[Scenario]


def lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    return_eigs: bool = False,
):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u

    Code from Mark Wilfred Mueller at http://www.mwm.im/lqr-controllers-with-python/

    Based on Bertsekas, p.151

    Yields the control law u = -K x
    """

    # first, try to solve the ricatti equation
    X = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    K = scipy.linalg.inv(R) @ (B.T @ X)

    if not return_eigs:
        return K
    else:
        eigVals, _ = scipy.linalg.eig(A - B * K)
        return K, eigVals
