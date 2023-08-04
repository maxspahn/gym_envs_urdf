"""Module for fsd sensor based on lidar."""
import numpy as np
import pybullet as p
import gymnasium as gym

from urdfenvs.sensors.lidar import Lidar
from urdfenvs.sensors.fsd_sensor import FSDSensor


class FreeSpaceDecompositionSensor(FSDSensor, Lidar):
    def __init__(
        self,
        link_name,
        max_radius: float = 1.0,
        number_constraints: int = 10,
        nb_rays: int = 10,
        ray_length: float = 10.0,
        angle_limits: np.ndarray = np.array([-np.pi, np.pi]),
        plotting_interval: int = -1,
        plotting_interval_fsd: int = -1,
        variance: float = 0.0,
        planar_visualization: bool = True,
    ):
        FSDSensor.__init__(
            self,
            max_radius,
            number_constraints=number_constraints,
            plotting_interval_fsd=plotting_interval_fsd,
            variance=variance,
            planar_visualization=planar_visualization,
        )
        Lidar.__init__(
            self,
            link_name,
            nb_rays=nb_rays,
            ray_length=ray_length,
            raw_data=False,
            angle_limits=angle_limits,
            variance=variance,
            plotting_interval=plotting_interval,
        )
        self._name = "FreeSpaceDecompSensor"

    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        lidar_observation = Lidar.sense(
            self, robot, obstacles, goals, t
        ).reshape((self._nb_rays, 2))
        lidar_position = np.array(p.getLinkState(robot, self._link_id)[0])
        relative_positions = np.concatenate(
            (
                np.reshape(lidar_observation, (self._nb_rays, 2)),
                np.zeros((self._nb_rays, 1)),
            ),
            axis=1,
        )
        self._height = lidar_position[2]
        absolute_positions = relative_positions + np.repeat(
            lidar_position[np.newaxis, :], self._nb_rays, axis=0
        )
        return self.compute_fsd(absolute_positions, lidar_position)
