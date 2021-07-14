"""DataModule for aggregating data points over a series of episodes, with additional
sampling from fixed sets.

Code based on the Pytorch Lightning example at
pl_examples/domain_templates/reinforce_learn_Qnet.py
"""
from typing import List, Callable, Tuple, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


from neural_clbf.systems import ControlAffineSystem


class EpisodicDataModule(pl.LightningDataModule):
    """
    DataModule for sampling from a replay buffer
    """

    def __init__(
        self,
        model: ControlAffineSystem,
        initial_domain: List[Tuple[float, float]],
        trajectories_per_episode: int = 100,
        trajectory_length: int = 5000,
        fixed_samples: int = 100000,
        max_points: int = 10000000,
        val_split: float = 0.1,
        batch_size: int = 64,
        quotas: Optional[Dict[str, float]] = None,
    ):
        """Initialize the DataModule

        args:
            model: the dynamics model to use in simulation
            initial_domain: the initial_domain to sample from, expressed as a list of
                             tuples denoting the min/max range for each dimension
            trajectories_per_episode: the number of rollouts to conduct at each episode
            trajectory_length: the number of samples to collect in each trajectory
            fixed_samples: the number of uniform samples to collect
            val_split: the fraction of sampled data to reserve for validation
            batch_size: the batch size
            quotas: a dictionary specifying the minimum percentage of the
                    fixed samples that should be taken from the safe,
                    unsafe, boundary, and goal sets. Expects keys to be either "safe",
                    "unsafe", "boundary", or "goal".
        """
        super().__init__()

        self.model = model
        self.n_dims = model.n_dims  # copied for convenience

        # Save the parameters
        self.trajectories_per_episode = trajectories_per_episode
        self.trajectory_length = trajectory_length
        self.fixed_samples = fixed_samples
        self.max_points = max_points
        self.val_split = val_split
        self.batch_size = batch_size
        if quotas is not None:
            self.quotas = quotas
        else:
            self.quotas = {}

        # Define the sampling intervals for initial conditions as a hyper-rectangle
        assert len(initial_domain) == self.n_dims
        self.initial_domain = initial_domain

        # Save the min, max, central point, and range tensors
        self.x_max, self.x_min = model.state_limits
        self.x_center = (self.x_max + self.x_min) / 2.0
        self.x_range = self.x_max - self.x_min

    def sample_trajectories(
        self, simulator: Callable[[torch.Tensor, int], torch.Tensor]
    ) -> torch.Tensor:
        """
        Generate new data points by simulating a bunch of trajectories

        args:
            simulator: a function that simulates the given initial conditions out for
                       the specified number of timesteps
        """
        # Start by sampling from initial conditions from the given region
        x_init = torch.Tensor(self.trajectories_per_episode, self.n_dims).uniform_(
            0.0, 1.0
        )
        for i in range(self.n_dims):
            min_val, max_val = self.initial_domain[i]
            x_init[:, i] = x_init[:, i] * (max_val - min_val) + min_val

        # Simulate each initial condition out for the specified number of steps
        x_sim = simulator(x_init, self.trajectory_length)

        # Reshape the data into a single replay buffer
        x_sim = x_sim.view(-1, self.n_dims)

        # Return the sampled data
        return x_sim

    def sample_fixed(self) -> torch.Tensor:
        """
        Generate new data points by sampling uniformly from the state space
        """
        samples = []
        # Figure out how many points are to be sampled at random, how many from the
        # goal, safe, or unsafe regions specifically
        allocated_samples = 0
        for region_name, quota in self.quotas.items():
            num_samples = int(self.fixed_samples * quota)
            allocated_samples += num_samples

            if region_name == "goal":
                samples.append(self.model.sample_goal(num_samples))
            elif region_name == "safe":
                samples.append(self.model.sample_safe(num_samples))
            elif region_name == "unsafe":
                samples.append(self.model.sample_unsafe(num_samples))
            elif region_name == "boundary":
                samples.append(self.model.sample_boundary(num_samples))

        # Sample all remaining points uniformly at random
        free_samples = self.fixed_samples - allocated_samples
        assert free_samples >= 0
        samples.append(self.model.sample_state_space(free_samples))

        return torch.vstack(samples)

    def prepare_data(self):
        """Create the dataset"""
        # Get some data points from simulations
        x_sim = self.sample_trajectories(self.model.nominal_simulator)

        # Augment those points with samples from the fixed range
        x_sample = self.sample_fixed()
        x = torch.cat((x_sim, x_sample), dim=0)

        # Randomly split data into training and test sets
        random_indices = torch.randperm(x.shape[0])
        val_pts = int(x.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]
        self.x_training = x[training_indices]
        self.x_validation = x[validation_indices]

        print("Full dataset:")
        print(f"\t{self.x_training.shape[0]} training")
        print(f"\t{self.x_validation.shape[0]} validation")
        print("\t----------------------")
        print(f"\t{self.model.goal_mask(self.x_training).sum()} goal points")
        print(f"\t({self.model.goal_mask(self.x_validation).sum()} val)")
        print(f"\t{self.model.safe_mask(self.x_training).sum()} safe points")
        print(f"\t({self.model.safe_mask(self.x_validation).sum()} val)")
        print(f"\t{self.model.unsafe_mask(self.x_training).sum()} unsafe points")
        print(f"\t({self.model.unsafe_mask(self.x_validation).sum()} val)")
        print(f"\t{self.model.boundary_mask(self.x_training).sum()} boundary points")
        print(f"\t({self.model.boundary_mask(self.x_validation).sum()} val)")

        # Turn these into tensor datasets
        self.training_data = TensorDataset(
            self.x_training,
            self.model.goal_mask(self.x_training),
            self.model.safe_mask(self.x_training),
            self.model.unsafe_mask(self.x_training),
        )
        self.validation_data = TensorDataset(
            self.x_validation,
            self.model.goal_mask(self.x_validation),
            self.model.safe_mask(self.x_validation),
            self.model.unsafe_mask(self.x_validation),
        )

    def add_data(self, simulator: Callable[[torch.Tensor, int], torch.Tensor]):
        """
        Augment the training and validation datasets by simulating and sampling

        args:
            simulator: a function that simulates the given initial conditions out for
                       the specified number of timesteps
        """
        print("\nAdding data!\n")
        # Get some data points from simulations
        x_sim = self.sample_trajectories(simulator)

        # # Augment those points with samples from the fixed range
        x_sample = self.sample_fixed()
        x = torch.cat((x_sim.type_as(x_sample), x_sample), dim=0)
        x = x.type_as(self.x_training)

        print(f"Sampled {x.shape[0]} new points")

        # Randomly split data into training and test sets
        random_indices = torch.randperm(x.shape[0])
        val_pts = int(x.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]

        print(f"\t{training_indices.shape[0]} train, {validation_indices.shape[0]} val")

        # Augment the existing data with the new points
        self.x_training = torch.cat((self.x_training, x[training_indices]))
        self.x_validation = torch.cat((self.x_validation, x[validation_indices]))

        # If we've exceeded the maximum number of points, forget the oldest
        if self.x_training.shape[0] + self.x_validation.shape[0] > self.max_points:
            print("Sample budget exceeded! Forgetting...")
            # Figure out how many training and validation points we should have
            n_val = int(self.max_points * self.val_split)
            n_train = self.max_points - n_val
            # And then keep only the most recent points
            self.x_training = self.x_training[-n_train:]
            self.x_validation = self.x_validation[-n_val:]

        print("Full dataset:")
        print(f"\t{self.x_training.shape[0]} training")
        print(f"\t{self.x_validation.shape[0]} validation")
        print("\t----------------------")
        print(f"\t{self.model.goal_mask(self.x_training).sum()} goal points")
        print(f"\t({self.model.goal_mask(self.x_validation).sum()} val)")
        print(f"\t{self.model.safe_mask(self.x_training).sum()} safe points")
        print(f"\t({self.model.safe_mask(self.x_validation).sum()} val)")
        print(f"\t{self.model.unsafe_mask(self.x_training).sum()} unsafe points")
        print(f"\t({self.model.unsafe_mask(self.x_validation).sum()} val)")

        # Save the new datasets
        self.training_data = TensorDataset(
            self.x_training,
            self.model.goal_mask(self.x_training),
            self.model.safe_mask(self.x_training),
            self.model.unsafe_mask(self.x_training),
        )
        self.validation_data = TensorDataset(
            self.x_validation,
            self.model.goal_mask(self.x_validation),
            self.model.safe_mask(self.x_validation),
            self.model.unsafe_mask(self.x_validation),
        )

    def setup(self, stage=None):
        """Setup -- nothing to do here"""
        pass

    def train_dataloader(self):
        """Make the DataLoader for training data"""
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        """Make the DataLoader for validation data"""
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=4,
        )
