"""Depth CNN/GRU in the same navigation plant used for component comparisons."""

from .pointcloud_navigation import PointCloudNavigationTask


class DepthNavigationTask(PointCloudNavigationTask):
    def __init__(self, config):
        super().__init__(config)
        self.observation_size = self.sensor.points_per_frame + self.observer.proprioception_size
