"""Plot data gathered for success and collision rates"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Define data, gathered from various scripts, in tidy data format

# PPO data gathered by running
# python scripts/test_policy.py \
#   data/2021-08-13_ppo_turtle2d/2021-08-13_15-23-36-ppo_turtle2d_s0 \
#   --len 100 --episodes 100 --norender
# in the safety_starter_agents directory, with the turtle2d env.
# Steps are converted to time with timestep 0.1
data = [
    {"Algorithm": "PPO", "Metric": "Goal-reaching rate", "Value": 44 / 100},
    {"Algorithm": "PPO", "Metric": "Collision rate", "Value": 11 / 100},
    {"Algorithm": "PPO", "Metric": "Time to goal ", "Value": 0.1 * float('nan')},
]
