"""Define an base class for a systems that yields observations"""
from abc import abstractmethod
from typing import Tuple, Optional, List

from matplotlib.axes import Axes
import numpy as np
from shapely.geometry import (
    box,
    GeometryCollection,
    LineString,
    Point,
    Polygon,
)
from shapely.affinity import rotate
import torch

from neural_clbf.systems.observable_system import ObservableSystem
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

    def add_walls(self, room_size: float) -> None:
        """Add walls to the scene (thin boxes)"""
        wall_width = 0.1
        semi_length = room_size / 2.0
        # Place the walls aligned with the x and y axes
        bottom_wall = box(
            -semi_length - wall_width,
            -semi_length - wall_width,
            semi_length + wall_width,
            -semi_length,
        )
        top_wall = box(
            -semi_length - wall_width,
            semi_length,
            semi_length + wall_width,
            semi_length + wall_width,
        )
        left_wall = box(
            -semi_length - wall_width,
            -semi_length - wall_width,
            -semi_length,
            semi_length + wall_width,
        )
        right_wall = box(
            semi_length,
            -semi_length - wall_width,
            semi_length + wall_width,
            semi_length + wall_width,
        )
        wall_obstacles = [bottom_wall, top_wall, left_wall, right_wall]

        # Add the obstacles
        for wall in wall_obstacles:
            self.add_obstacle(wall)

    def add_random_box(
        self,
        size_range: Tuple[float, float],
        x_position_range: Tuple[float, float],
        y_position_range: Tuple[float, float],
        rotation_range: Tuple[float, float],
    ) -> None:
        """Add a random box to the scene

        args:
            size_range: tuple of min and max side lengths
            x_position_range: tuple of min and max positions for center (in x)
            y_position_range: tuple of min and max positions for center (in y)
            rotation_range: tuple of min and max rotations
        """
        # Build the box without rotation
        semi_height = np.random.uniform(*size_range) / 2.0
        semi_width = np.random.uniform(*size_range) / 2.0
        center_x = np.random.uniform(*x_position_range)
        center_y = np.random.uniform(*y_position_range)

        lower_left_x = center_x - semi_width
        lower_left_y = center_y - semi_height
        upper_right_x = center_x + semi_width
        upper_right_y = center_y + semi_height

        new_box = box(lower_left_x, lower_left_y, upper_right_x, upper_right_y)

        # Add some rotation
        rotation_angle = np.random.uniform(*rotation_range)
        rotated_box = rotate(new_box, rotation_angle, use_radians=True)

        # Add the box to the scene
        self.add_obstacle(rotated_box)

    def add_random_boxes(
        self,
        num_boxes: int,
        size_range: Tuple[float, float],
        x_position_range: Tuple[float, float],
        y_position_range: Tuple[float, float],
        rotation_range: Tuple[float, float],
    ) -> None:
        """Add random boxes to the scene

        args:
            num_boxes: how many boxes to add
            size_range: tuple of min and max side lengths
            x_position_range: tuple of min and max positions for center (in x)
            y_position_range: tuple of min and max positions for center (in y)
            rotation_range: tuple of min and max rotations
        """
        for _ in range(num_boxes):
            self.add_random_box(
                size_range, x_position_range, y_position_range, rotation_range
            )

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

    def lidar_measurement(
        self,
        qs: torch.Tensor,
        num_rays: int = 32,
        field_of_view: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
        max_distance: float = 100,
        noise: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a simulated LIDAR measurement of the scene, taken from the specified pose

        args:
            qs: a N x 6 tensor containing the x, y, and theta coordinates for each of N
                measurements to be taken, along with the velocity in each coordinate.
            num_rays: the number of equally spaced rays to measure
            field_of_view: a tuple specifying the maximum and minimum angle of the field
                           of view of the LIDAR sensor, measured in the vehicle frame
            max_distance: Any rays that would measure a greater distance will not
                          register a contact.
            noise: if non-zero, apply white Gaussian noise with this standard deviation
                   and zero mean to all measurements.
        returns:
            an N x num_rays x 4 tensor containing the measurements along each
                ray. Rays are ordered in the counter-clockwise direction, and each
                measurement contains the (x, y) location of the contact point and the
                velocity (xdot, ydot) (as might be obtained by subtracting the last
                measurement). These measurements will be in the agent frame. If no
                contact point is found within max_distance, then a 0.0 will be placed
                in the corresponding element of the second return tensor (a 1 otherwise)
            an N x num_rays tensor indicating which data points are valid
        """
        # Sanity check on inputs
        assert field_of_view[1] > field_of_view[0], "Field of view must be (min, max)"
        # Reshape input if necessary
        if qs.ndim == 1:
            qs = torch.reshape(qs, (1, -1))

        # Create the array to store the results, then iterate through each sample point
        measurements = torch.zeros(qs.shape[0], 4, num_rays).type_as(qs)
        valid = torch.ones(qs.shape[0], num_rays).type_as(qs)

        # Figure out the angles to measure on
        angles = torch.linspace(field_of_view[0], field_of_view[1], num_rays)

        for q_idx, q in enumerate(qs):
            agent_point = Point(q[0].item(), q[1].item())
            # Sweep through the field of view, checking for an intersection
            # out to max_distance
            for ray_idx in range(num_rays):
                ray_start = q[:2].detach().numpy()  # start at the agent (x, y)
                ray_direction = np.array(
                    [
                        np.cos(q[2].detach().item() + angles[ray_idx]),
                        np.sin(q[2].detach().item() + angles[ray_idx]),
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
                    # Figure out which point is closest
                    closest_idx = np.argmin(
                        [
                            agent_point.distance(intersection)
                            for intersection in intersections
                        ]
                    )
                    contact_pt = intersections[closest_idx]  # type: ignore
                else:
                    # If no intersection was found, mark the corresponding element
                    # of the valid mask tensor and continue
                    valid[q_idx, ray_idx] = 0.0
                    continue

                # Get the coordinates of that point
                contact_x, contact_y = contact_pt.coords[0]

                # Add noise if necessary
                if noise > 0:
                    contact_x += np.random.normal(0.0, noise)
                    contact_y += np.random.normal(0.0, noise)

                # Get the point relative to the agent coordinates in the world frame
                contact_pt_world = torch.tensor([contact_x, contact_y]).type_as(q)
                contact_offset_world = contact_pt_world - q[:2]
                # Rotate the point by -theta to bring it into the agent frame
                rotation_mat = torch.tensor(
                    [
                        [torch.cos(q[2]), -torch.sin(q[2])],
                        [torch.sin(q[2]), torch.cos(q[2])],
                    ]
                )
                contact_pt_agent = torch.matmul(rotation_mat, contact_offset_world)

                # Now we need to get the velocity, which we can derive by hand
                theta = q[2]
                s_theta = torch.sin(theta)
                c_theta = torch.cos(theta)
                theta_dot = q[5]
                vx = q[3]
                vy = q[4]
                contact_vx_agent = (
                    -theta_dot * s_theta * contact_offset_world[0]
                    - c_theta * vx
                    - theta_dot * c_theta * contact_offset_world[1]
                    + s_theta * vy
                )
                contact_vy_agent = (
                    theta_dot * c_theta * contact_offset_world[0]
                    - s_theta * vx
                    - theta_dot * s_theta * contact_offset_world[1]
                    - c_theta * vy
                )

                # Save the measurement (we've already marked if this point is valid)
                measurements[q_idx, :2, ray_idx] = contact_pt_agent
                measurements[q_idx, 2, ray_idx] = contact_vx_agent
                measurements[q_idx, 3, ray_idx] = contact_vy_agent

        return measurements, valid

    def plot(self, ax: Axes):
        """Plot the given scene

        args:
            ax: the matplotlib Axes on which to plot
        """
        # Plotting the scene is as simple as plotting each obstacle
        for obstacle in self.obstacles:
            x_pts, y_pts = obstacle.exterior.xy
            ax.fill(x_pts, y_pts, alpha=0.3, fc="k", ec="none")


class PlanarLidarSystem(ObservableSystem):
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
        noise: float = 0.0,
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
            noise: the standard deviation of gaussian noise to apply to observations
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
        self.noise = noise

    @abstractmethod
    def planar_configuration(self, x: torch.Tensor) -> torch.Tensor:
        """Get the x and y position and orientation of this agent in the 2D plane

        args:
            x: an n x self.n_dims tensor of state

        returns:
            an n x 3 tensor of [x, y, theta]
        """
        pass

    @property
    def n_obs(self) -> int:
        return self.num_rays

    @property
    def obs_dim(self) -> int:
        return 4

    def get_observations(self, x: torch.Tensor) -> torch.Tensor:
        """Get the vector of measurements at this point

        args:
            x: an N x self.n_dims tensor of state

        returns:
            an N x self.obs_dim x self.n_obs tensor containing the observed data
        """
        # Get the lidar measurements from the scene
        qs = self.planar_configuration(x)
        measurements, _ = self.scene.lidar_measurement(
            qs,
            num_rays=self.num_rays,
            field_of_view=self.field_of_view,
            max_distance=self.max_distance,
            noise=self.noise,
        )

        return measurements

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating safe regions for this system

        args:
            x: a tensor of points in the state space
        """
        # A state is safe if the closest lidar point is at least some distance away
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)
        min_safe_ray_length = 0.5

        measurements = self.get_observations(x)

        # Get the x and y values
        ray_lengths = measurements[:, :2, :].norm(dim=1)

        safe_mask.logical_and_(ray_lengths.min(dim=1)[0] >= min_safe_ray_length)

        return safe_mask

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating unsafe regions for this system

        args:
            x: a tensor of points in the state space
        """
        # A state is safe if the closest lidar point too close
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)
        min_safe_ray_length = 0.1

        measurements = self.get_observations(x)

        # Get the x and y values
        ray_lengths = measurements[:, :2, :].norm(dim=1)

        unsafe_mask.logical_or_(ray_lengths.min(dim=1)[0] <= min_safe_ray_length)

        return unsafe_mask

    def plot_environment(self, ax: Axes) -> None:
        """
        Add a plot of the environment to the given figure by plotting the underlying
        scene.

        args:
            ax: the axis on which to plot
        """
        self.scene.plot(ax)
