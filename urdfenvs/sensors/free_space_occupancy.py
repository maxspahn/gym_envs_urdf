"""Module for fsd sensor based on lidar."""
import numpy as np
import pybullet as p
import gymnasium as gym

from urdfenvs.sensors.fsd_sensor import FSDSensor
from urdfenvs.sensors.occupancy_sensor import OccupancySensor
from urdfenvs.urdf_common.helpers import extract_link_id


class FreeSpaceOccupancySensor(FSDSensor, OccupancySensor):
    def __init__(
        self,
        link_name,
        name: str = "FreeSpaceDecompSensor",
        max_radius: float = 1.0,
        number_constraints: int = 10,
        limits: np.ndarray = np.array([[-1, 1], [-1, 1], [-1, 1]]),
        resolution: np.ndarray = np.array([10, 10, 10], dtype=int),
        interval: int = -1,
        variance: float = 0.0,
        plotting_interval: int = -1,
        plotting_interval_fsd: int = -1,
        planar_visualization: bool = False,
    ):
        FSDSensor.__init__(
            self,
            max_radius,
            number_constraints=number_constraints,
            plotting_interval_fsd=plotting_interval_fsd,
            planar_visualization=planar_visualization,
            variance=variance,
        )
        OccupancySensor.__init__(
            self,
            limits=limits,
            resolution=resolution,
            interval=interval,
            variance=variance,
            plotting_interval=plotting_interval,
        )
        self._link_name = link_name
        self._name = name
        self._height = 0.3

    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        occupancy = OccupancySensor.sense(
            self, robot, obstacles, goals, t
        ).reshape((-1, 1))
        link_id = extract_link_id(robot, self._link_name)
        seed_point = np.array(p.getLinkState(robot, link_id)[0])
        point_positions = self._mesh_flat[np.where(occupancy == 1)[0]]
        return self.compute_fsd(point_positions, seed_point)
