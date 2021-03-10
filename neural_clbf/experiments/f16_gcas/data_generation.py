"""File for generating training data for the F16 GCAS experiment"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, random_split

from neural_clbf.systems.f16 import F16


class F16GcasDataModule(pl.LightningDataModule):
    """
    DataModule for generating sample points (and a corresponding safe/unsafe mask) for
    the 2D quadrotor obstacle avoidance experiment.

    # TODO @dawsonc move this to a common superclass
    """

    def __init__(
        self,
        N_samples: int = 5000000,
        domains: Optional[List[List[Tuple[float, float]]]] = None,
        split: float = 0.1,
        batch_size: int = 64,
    ):
        """Initialize the DataModule

        args:
            N_samples: the number of points to sample
            domains: a list of domains to sample from, where a domain is a list of
                     tuples denoting the min/max range for each dimension
            split: the fraction of sampled data to reserve for validation
            batch_size: the batch size
        """
        super().__init__()

        # Define the sample and batch sizes
        self.N_samples = N_samples
        self.split = split
        self.batch_size = batch_size

        # Define the sampling intervals
        # (we allow for multiple domains in case you need to sample extra around
        # the origin, for example)
        self.n_dims = F16.N_DIMS
        if domains is None:
            self.domains = [
                [
                    (400, 600),  # vt
                    (-np.pi, np.pi),  # alpha
                    (-np.pi, np.pi),  # beta
                    (-np.pi, np.pi),  # phi
                    (-np.pi, np.pi),  # theta
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
                ],
            ]
        else:
            for domain in domains:
                assert len(domain) == self.n_dims
            self.domains = domains

        # Since we sample N_samples from each domain, the total number of samples
        # might be more than just N_samples
        self.N = len(self.domains) * N_samples

    def safe_mask_fn(self, x):
        """Return the mask of x indicating safe regions for the GCAS

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # GCAS activates under 1000 feet
        safe_height = 1000
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
        nose_high_enough = x[:, F16.THETA] + x[:, F16.ALPHA] >= 0.0
        goal_mask.logical_and_(nose_high_enough)
        roll_rate_low = x[:, F16.P].abs() <= 0.1745  # 10 degrees to radians
        goal_mask.logical_and_(roll_rate_low)
        wings_near_level = x[:, F16.PHI].abs() <= 0.0873  # 5 degrees to radians
        goal_mask.logical_and_(wings_near_level)
        above_deck = x[:, F16.H] >= 1000.0
        goal_mask.logical_and_(above_deck)

        return goal_mask

    def prepare_data(self):
        """Create the dataset by randomly sampling from the domains"""
        # Loop through each domain, sample from it, and store the samples in this list
        x_samples = []
        for domain in self.domains:
            # Sample all dimensions from [0, 1], then scale and shift as needed
            x_sample = torch.Tensor(self.N_samples, self.n_dims).uniform_(0.0, 1.0)
            for i in range(self.n_dims):
                min_val, max_val = domain[i]
                x_sample[:, i] = x_sample[:, i] * (max_val - min_val) + min_val

            # Save the sample from this domain
            x_samples.append(x_sample)

        # Concatenate all the samples into one big tensor
        x = torch.vstack(x_samples)

        # Create the goal mask for the sampled points
        goal_mask = self.goal_mask_fn(x)

        # Create the safe/unsafe masks for the sampled points
        safe_mask = self.safe_mask_fn(x)
        unsafe_mask = self.unsafe_mask_fn(x)

        # Combine all of these into a single dataset and save it
        self.dataset = TensorDataset(x, goal_mask, safe_mask, unsafe_mask)

    def setup(self, stage=None):
        """Setup the data for training and validation"""
        # The data were generated randomly, so no need to do a random split
        val_pts = int(self.N * self.split)
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
