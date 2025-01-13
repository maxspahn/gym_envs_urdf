import numpy as np

from urdfenvs.sensors.pybullet.fsd_sensor import FSDSensorPybullet
from urdfenvs.sensors.pybullet.lidar import LidarPybullet


class FreeSpaceDecompositionSensorPybullet(FSDSensorPybullet, LidarPybullet):
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
        physics_engine_name: str = 'pybullet'
    ):
        FSDSensorPybullet.__init__(
            self,
            max_radius,
            number_constraints=number_constraints,
            plotting_interval_fsd=plotting_interval_fsd,
            variance=variance,
            planar_visualization=planar_visualization,
            physics_engine_name=physics_engine_name,
        )
        LidarPybullet.__init__(
            self,
            link_name,
            nb_rays=nb_rays,
            ray_length=ray_length,
            raw_data=False,
            angle_limits=angle_limits,
            variance=variance,
            plotting_interval=plotting_interval,
            physics_engine_name=physics_engine_name,
        )
        self._name = "FreeSpaceDecompSensor"

    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        lidar_observation = LidarPybullet.sense(
            self, robot, obstacles, goals, t
        ).reshape((self._nb_rays, 2))
        lidar_position = self._physics_engine.get_link_position(robot, self._link_id)
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
