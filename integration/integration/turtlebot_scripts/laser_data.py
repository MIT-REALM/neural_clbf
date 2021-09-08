import torch
from scipy import interpolate


class LidarMonitor(object):
    """A class to monitor lidar data and save the most recent set"""

    def __init__(
        self,
        num_rays: int = 32,
    ):
        super(LidarMonitor, self).__init__()
        self.num_rays = num_rays

        # Create a place to store the sensor data in between callbacks
        self.last_scan = torch.zeros(1, 2, num_rays)  # x and y in local frame

    def scan_callback(self, msg):
        # Make a tensor of angles
        angles = torch.arange(
            msg.angle_min, msg.angle_max + msg.angle_increment, msg.angle_increment
        )
        # Make a tensor of ranges
        ranges = torch.tensor(msg.ranges)
        # By default, rays that don't make contact are set to 0, which we don't want
        # So reset to max_distance
        ranges[ranges < msg.range_min] = msg.range_max
        # Adjust by a bit of buffer
        ranges -= 0.08

        # Downsample to the correct number of rays
        ray_angles = torch.linspace(msg.angle_min, msg.angle_max, self.num_rays)
        ray_ranges = torch.tensor(interpolate.interp1d(angles, ranges)(ray_angles))

        # Convert to cartesian points
        x_coords = ray_ranges * torch.cos(ray_angles)
        y_coords = ray_ranges * torch.sin(ray_angles)
        self.last_scan[0, :, :] = torch.cat(
            (x_coords.view(1, -1), y_coords.view(1, -1)), dim=0
        )
