"""Defines useful constants and helper functions for dynamical systems"""
from typing import Dict, List


# Gravitation acceleration
grav = 9.80665

# Define a type alias for parameter scenarios
Scenario = Dict[str, float]
ScenarioList = List[Scenario]
