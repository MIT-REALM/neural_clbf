"""File for generating training data for the F16 GCAS experiment"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, random_split

from neural_clbf.systems.f16 import F16


class F16GcasSimDataModule(pl.LightningDataModule):
    """
    DataModule for generating sample points (and a corresponding safe/unsafe mask) for
    the 2D quadrotor obstacle avoidance experiment.

    Generates random data by simulating forward from random initial conditions.

    # TODO @dawsonc move this to a common superclass
    """

    def __init__(
        self,
        f16_model: F16,
        N_samples: int = 5000000,
        sim_steps: int = 1000,
        timestep: float = 0.01,
        initial_domains: Optional[List[List[Tuple[float, float]]]] = None,
        state_limits: Optional[List[Tuple[float, float]]] = None,
        val_split: float = 0.1,
        batch_size: int = 64,
    ):
        """Initialize the DataModule

        args:
            f16_model: the dynamics model to use in simulation
            N_samples: the number of points to sample from each domain
            sim_steps: the number of steps to sample for each simulation
            timestep: the simulation timestep
            initial_domains: a list of initial_domains to sample from, where a domain is
                             a list of tuples denoting the min/max range for each
                             dimension
            state_limits: the expected maximum and minimum value for each state dimsnion
            val_split: the fraction of sampled data to reserve for validation
            batch_size: the batch size
        """
        super().__init__()

        self.f16_model = f16_model
        self.delta_t = timestep

        # Define the sample and batch sizes
        self.N_samples = N_samples
        self.sim_steps = sim_steps
        self.val_split = val_split
        self.batch_size = batch_size

        # Define the limits on states
        self.n_dims = F16.N_DIMS
        if state_limits is None:
            self.state_limits = [
                (400, 600),  # vt
                (-1.0, 1.0),  # alpha
                (-1.0, 1.0),  # beta
                (-np.pi / 2.0, np.pi / 2.0),  # phi
                (-np.pi / 2.0, np.pi / 2.0),  # theta
                (-np.pi, np.pi),  # psi
                (-2 * np.pi, 2 * np.pi),  # P
                (-2 * np.pi, 2 * np.pi),  # Q
                (-2 * np.pi, 2 * np.pi),  # R
                (-1000, 1000),  # pos_n
                (-1000, 1000),  # pos_e
                (0.0, 1500),  # alt
                (0, 10),  # pow
                (-20, 20),  # nz_int
                (-20, 20),  # ps_int
                (-20, 20),  # nyr_int
            ]
        else:
            assert len(state_limits) == self.n_dims
            self.state_limits = state_limits

        # Define the sampling intervals
        # (we allow for multiple initial_domains in case you need to sample extra around
        # the origin, for example)
        if initial_domains is None:
            self.initial_domains = [
                [
                    (400, 600),  # vt
                    (-1.0, 1.0),  # alpha
                    (-1.0, 1.0),  # beta
                    (-np.pi / 4.0, np.pi / 4.0),  # phi
                    (-np.pi / 4.0, np.pi / 4.0),  # theta
                    (-np.pi / 4.0, np.pi / 4.0),  # psi
                    (-np.pi, np.pi),  # P
                    (-np.pi, np.pi),  # Q
                    (-np.pi, np.pi),  # R
                    (-100.0, 100.0),  # pos_n
                    (-100.0, 100.0),  # pos_e
                    (500.0, 1500),  # alt
                    (0, 10),  # pow
                    (0.0, 0.0),  # nz_int
                    (0.0, 0.0),  # ps_int
                    (0.0, 0.0),  # nyr_int
                ],
            ]
        else:
            for domain in initial_domains:
                assert len(domain) == self.n_dims
            self.initial_domains = initial_domains

        # Since we sample N_samples from each domain, the total number of samples
        # might be more than just N_samples
        self.N_sims = N_samples // sim_steps
        self.N = len(self.initial_domains) * self.N_sims * sim_steps

        # Save the min, max, central point, and range tensors
        self.x_min = torch.tensor([lim[0] for lim in self.state_limits])
        self.x_max = torch.tensor([lim[1] for lim in self.state_limits])
        self.x_center = (self.x_max + self.x_min) / 2.0
        self.x_center = (self.x_max + self.x_min) / 2.0
        self.x_range = self.x_max - self.x_min

    def state_limit_mask_fn(self, x):
        """Return the mask of x indicating points within the given state limits

        args:
            x: a tensor of points in the state space
        """
        in_limit_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        for i in range(self.n_dims):
            under_max = x[:, i] <= self.state_limits[i][1]
            over_min = x[:, i] >= self.state_limits[i][0]
            in_limit_mask.logical_and_(under_max)
            in_limit_mask.logical_and_(over_min)

        return in_limit_mask

    def safe_mask_fn(self, x):
        """Return the mask of x indicating safe regions for the GCAS

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # GCAS activates under 1000 feet
        safe_height = 900
        floor_mask = x[:, F16.H] >= safe_height
        safe_mask.logical_and_(floor_mask)

        return safe_mask

    def unsafe_mask_fn(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # We have a floor that we need to avoid
        unsafe_height = 500
        floor_mask = x[:, F16.H] <= unsafe_height
        unsafe_mask.logical_or_(floor_mask)

        return unsafe_mask

    def goal_mask_fn(self, x):
        """Return the mask of x indicating points in the goal set (within 0.2 m of the
        goal).

        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as anywhere where the aircraft is near level and above
        # the deck
        nose_high_enough = x[:, F16.THETA] + x[:, F16.ALPHA] <= 0.0
        goal_mask.logical_and_(nose_high_enough)
        roll_rate_low = x[:, F16.P].abs() <= 0.25
        goal_mask.logical_and_(roll_rate_low)
        wings_near_level = x[:, F16.PHI].abs() <= 0.1
        goal_mask.logical_and_(wings_near_level)
        above_deck = x[:, F16.H] >= 1000.0
        goal_mask.logical_and_(above_deck)

        return goal_mask

    def prepare_data(self):
        """Create the dataset by randomly sampling from the initial_domains"""
        # Loop through each domain, sample from it, and store the samples in this list
        # to be used later as initial conditions for the simulation
        # Sample all dimensions from [0, 1], then scale and shift as needed
        x_init = torch.Tensor(
            len(self.initial_domains), self.N_sims, self.n_dims
        ).uniform_(0.0, 1.0)
        for domain_idx, domain in enumerate(self.initial_domains):
            for i in range(self.n_dims):
                min_val, max_val = domain[i]
                x_init[domain_idx, :, i] = (
                    x_init[domain_idx, :, i] * (max_val - min_val) + min_val
                )

        # Make a tensor to hold the simulation results (all nan for now, until filled
        # with real data)
        x = (
            torch.zeros(
                (len(self.initial_domains), self.N_sims, self.sim_steps, self.n_dims)
            )
            + np.nan
        )
        for domain_idx, domain in enumerate(self.initial_domains):
            for sim_idx in range(self.N_sims):
                # Load in the initial condition
                x[domain_idx, sim_idx, 0, :] = x_init[domain_idx, sim_idx, :]

                # Simulate until we finish the simulation or leave the state limits
                for tstep in range(1, self.sim_steps):
                    x_current = x[domain_idx, sim_idx, tstep - 1, :]

                    # End the simulation if we leave the state limits
                    if not self.state_limit_mask_fn(x_current.unsqueeze(0)).all():
                        break

                    # Otherwise, get the nominal controller and step forward
                    u = self.f16_model.u_nominal(x_current.unsqueeze(0))
                    xdot = self.f16_model.closed_loop_dynamics(
                        x_current.unsqueeze(0), u
                    )
                    x[domain_idx, sim_idx, tstep, :] = (
                        x_current + self.delta_t * xdot.squeeze()
                    )

        # Reshape the simulations into one replay buffer
        x = x.view(-1, self.n_dims)
        # Remove all nans
        x = x[~torch.any(x.isnan(), dim=-1)]
        # Recalculate how many samples we have
        self.N = x.shape[0]

        # Create the goal mask for the sampled points
        goal_mask = self.goal_mask_fn(x)

        # Create the safe/unsafe masks for the sampled points
        safe_mask = self.safe_mask_fn(x)
        unsafe_mask = self.unsafe_mask_fn(x)

        # Combine all of these into a single dataset and save it
        self.dataset = TensorDataset(x, goal_mask, safe_mask, unsafe_mask)

    def setup(self, stage=None):
        """Setup the data for training and validation"""
        # The data were generated randomly, so no need to do a random val_split
        val_pts = int(self.N * self.val_split)
        split_lens = [self.N - val_pts, val_pts]
        self.train_data, self.validation_data = random_split(self.dataset, split_lens)

    def train_dataloader(self):
        """Make the DataLoader for training data"""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=8,
        )

    def val_dataloader(self):
        """Make the DataLoader for validation data"""
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=8,
        )
