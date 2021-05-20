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
    """Solve the discrete time lqr controller.

    x_{t+1} = A x_t + B u_t

    cost = sum x.T*Q*x + u.T*R*u

    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    http://www.mwm.im/lqr-controllers-with-python/

    Based on Bertsekas, p.151

    Yields the control law u = -K x
    """

    # first, try to solve the ricatti equation
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = scipy.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    if not return_eigs:
        return K
    else:
        eigVals, _ = scipy.linalg.eig(A - B * K)
        return K, eigVals


def continuous_lyap(Acl: np.ndarray, Q: np.ndarray):
    """Solve the continuous time lyapunov equation.

    Acl.T P + P Acl + Q = 0

    using scipy, which expects AP + PA.T = Q, so we need to transpose Acl and negate Q
    """
    P = scipy.linalg.solve_continuous_lyapunov(Acl.T, -Q)
    return P


def discrete_lyap(Acl: np.ndarray, Q: np.ndarray):
    """Solve the continuous time lyapunov equation.

    Acl.T P Acl - P + Q = 0

    using scipy, which expects A P A.T - P + Q = 0, so we need to transpose Acl
    """
    P = scipy.linalg.solve_discrete_lyapunov(Acl.T, Q)
    return P
