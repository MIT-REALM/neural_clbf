"""Define an base class for a systems that yields observations"""
from abc import abstractmethod
from typing import Tuple, Optional, List

from matplotlib.axes import Axes
import numpy as np
from shapely.geometry import (
    GeometryCollection,
    LineString,
    Point,
    Polygon,
)
import torch

from neural_clbf.systems.control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import (
    Scenario,
    ScenarioList,
)


class Scene:
    """
    Represents a 2D scene of polygonal obstacles
    """

    def __init__(self, obstacles: List[Polygon]):
        """Initialize a scene containing the specified obstacles

        args:
            obstacles: a list of `shapely.Polygon`s representing the obstacles in the
                       scene
        """
        # Save the provided obstacles
        self.obstacles = obstacles

    def add_obstacle(self, obstacle: Polygon) -> None:
        """Add an obstacle to the scene

        args:
            obstacle: a `shapely.Polygon` representing the obstacle to be added
        """
        if obstacle not in self.obstacles:
            self.obstacles.append(obstacle)

    def remove_obstacle(self, obstacle: Polygon) -> None:
        """Remove an obstacle from the scene

        args:
            obstacle: a `shapely.Polygon` representing the obstacle to be removed
        raises:
            ValueError if the specified obstacle is not in the scene
        """
        self.obstacles.remove(
            obstacle
        )  # will raise a ValueError if obstacle is not in obstacles

    @torch.no_grad()
    def lidar_measurement(
        self,
        qs: torch.Tensor,
        num_rays: int = 32,
        field_of_view: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
        max_distance: float = 100,
        noise: float = 0.0,
    ) -> torch.Tensor:
        """Return a simulated LIDAR measurement of the scene, taken from the specified pose

        args:
            qs: a Nx3 tensor containing the x, y, and theta coordinates for each of N
                measurements to be taken.
            num_rays: the number of equally spaced rays to measure
            field_of_view: a tuple specifying the maximum and minimum angle of the field
                           of view of the LIDAR sensor, measured in the vehicle frame
            max_distance: Any rays that would measure a greater distance will saturate
                          at this value.
            noise: if non-zero, apply white Gaussian noise with this standard deviation
                   and zero mean to all measurements.
        returns:
            an N x num-rays tensor containing the measurements along each ray. Rays are
            ordered in the counter-clockwise direction.
        """
        # Sanity check on inputs
        assert field_of_view[1] > field_of_view[0], "Field of view must be (min, max)"
        # Reshape input if necessary
        if qs.ndim == 1:
            qs = torch.reshape(qs, (1, -1))

        # Create the array to store the results, then iterate through each sample point
        # We initialize each measurement to max_distance, since that's the desired value
        # if we don't find an intersection
        measurements = np.zeros((qs.shape[0], num_rays)) + max_distance

        for q_idx, q in enumerate(qs):
            agent_point = Point(q[0].item(), q[1].item())
            # Sweep through the field of view, checking for an intersection
            # out to max_distance
            sweep_angle = float(field_of_view[1] - field_of_view[0]) / num_rays
            for ray_idx in range(num_rays):
                ray_start = q[:2].numpy()  # start at the agent (x, y)
                ray_direction = np.array(
                    [
                        np.cos(q[2].item() + field_of_view[0] + ray_idx * sweep_angle),
                        np.sin(q[2].item() + field_of_view[0] + ray_idx * sweep_angle),
                    ]
                )
                ray_end = ray_start + max_distance * ray_direction
                ray = LineString([ray_start, ray_end])

                # Find nearest intersection with the scene
                intersections = []
                for obstacle in self.obstacles:
                    # Skip obstacles that don't touch the ray
                    if not obstacle.intersects(ray):
                        continue
                    # Otherwise, save the intersections
                    # The intersection could either be a single object, or a collection,
                    # which needs to be dealt with separately
                    current_intersections = obstacle.intersection(ray)
                    if isinstance(current_intersections, GeometryCollection):
                        intersections += current_intersections.geoms
                    else:
                        intersections.append(current_intersections)

                # Get the nearest distance and save that as the measurement
                # (with noise if needed)
                if intersections:
                    measurements[q_idx, ray_idx] = min(
                        [
                            agent_point.distance(intersection)
                            for intersection in intersections
                        ]
                    )
                    if noise > 0:
                        measurements[q_idx, ray_idx] += np.random.normal(0.0, noise)

        return torch.tensor(measurements).type_as(qs)

    def plot_scene(self, ax: Axes):
        """Plot the given scene

        args:
            ax: the matplotlib Axes on which to plot
        """
        # Plotting the scene is as simple as plotting each obstacle
        for obstacle in self.obstacles:
            x_pts, y_pts = obstacle.exterior.xy
            ax.fill(x_pts, y_pts, alpha=0.3, fc="k", ec="none")


class PlanarLidarSystem(ControlAffineSystem):
    """
    Represents a generic dynamical system that lives in a plane and observes its
    environment via Lidar.
    """

    def __init__(
        self,
        nominal_params: Scenario,
        scene: Scene,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        use_linearized_controller: bool = True,
        scenarios: Optional[ScenarioList] = None,
        num_rays: int = 10,
        field_of_view: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
        max_distance: float = 10.0,
    ):
        """
        Initialize a system.

        args:
            nominal_params: a dictionary giving the parameter values for the system
            scene: the 2D scene that the system inhabits.
            dt: the timestep to use for simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
            use_linearized_controller: if True, linearize the system model to derive a
                                       LQR controller. If false, the system is must
                                       set self.P itself to be a tensor n_dims x n_dims
                                       positive definite matrix.
            scenarios: an optional list of scenarios for robust control
            num_rays: the number of Lidar rays
            field_of_view: the minimum and maximum angle at which to take lidar rays
            max_distance: lidar saturation distance
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super(PlanarLidarSystem, self).__init__(
            nominal_params=nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            use_linearized_controller=use_linearized_controller,
            scenarios=scenarios,
        )

        # Save the provided scene
        self.scene = scene

        # Save other parameters
        self.num_rays = num_rays
        self.field_of_view = field_of_view
        self.max_distance = max_distance

    @abstractmethod
    def planar_configuration(self, x: torch.Tensor) -> torch.Tensor:
        """Get the x and y position and orientation of this agent in the 2D plane

        args:
            x: an n x self.n_dims tensor of state

        returns:
            an n x 3 tensor of [x, y, theta]
        """
        pass

    def get_observations(self, x: torch.Tensor) -> torch.Tensor:
        """Get the vector of lidar measurements at this point

        args:
            x: an n x self.n_dims tensor of state

        returns:
            an n x self.num_rays tensor of lidar distance measurements
        """
        measurements = torch.zeros(x.shape[0], self.num_rays).type_as(x)

        # We can only query the scene at one point at a time, so loop through x
        for idx, x_row in enumerate(x):
            q = self.planar_configuration(x_row)
            measurements[idx, :] = self.scene.lidar_measurement(
                q, self.num_rays, self.field_of_view, self.max_distance
            )

        return measurements

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating safe regions for this system

        args:
            x: a tensor of points in the state space
        """
        # A state is safe if the shortest lidar ray is at least some distance long.
        safe_mask = torch.ones_like(x, dtype=torch.bool)
        min_safe_ray_length = 0.25

        measurements = self.get_observations(x)

        safe_mask.logical_and_(measurements.min(dim=1)[0] >= min_safe_ray_length)

        return safe_mask

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating unsafe regions for this system

        args:
            x: a tensor of points in the state space
        """
        # A state is unsafe if it is in collision with any obstacle in the scene
        unsafe_mask = torch.zeros_like(x, dtype=torch.bool)
        min_safe_ray_length = 0.1

        measurements = self.get_observations(x)

        unsafe_mask.logical_or_(measurements.min(dim=1)[0] <= min_safe_ray_length)

        return unsafe_mask
