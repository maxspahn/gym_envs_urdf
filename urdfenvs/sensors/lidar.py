"""Module for lidar sensor simulation."""
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation

from urdfenvs.sensors.sensor import Sensor

class Lidar(Sensor):
    """
    The Lidar sensor senses the distance toward the next object. A maximum
    sensing distance and the number of rays can be set. The Rays are evenly
    distributed in a circle.

    Attributes
    ----------

    _nb_rays: int
        Number of lidar rays spread over 2 pi.
    _ray_length: float
        Length of a single ray, maximum detection distance.
    _link_id: int
        Link of robot to which the lidar is connected.
    _thetas: list
        Angles for which rays are emitted.
    _rel_positions: np.ndarray
        Relative positions of first obstacle for each ray (x, y).
    _raw_data: bool
        Switch whether relative positions or raw distances are returned.
    _distance: np.ndarray
        Raw distance information for rays.
    """

    def __init__(
        self,
        link_name,
        nb_rays=10,
        ray_length=10.0,
        raw_data=True,
        angle_limits: np.ndarray = np.array([-np.pi, np.pi]),
        plotting_interval: int = -1,
        variance: float = 0.0,
        physics_engine_name: str = 'pybullet',
    ):
        super().__init__(
            "LidarSensor",
            variance=variance,
            plotting_interval=plotting_interval,
            physics_engine_name=physics_engine_name,
        )
        self._nb_rays = nb_rays
        self._raw_data = raw_data
        self._ray_length = ray_length
        self._link_name = link_name
        self._link_id = None
        if isinstance(link_name, int):
            self._link_id = link_name
        self._angle_limits = angle_limits
        self._thetas = [
            angle_limits[0]
            + i * (angle_limits[1] - angle_limits[0]) / self._nb_rays
            for i in range(self._nb_rays)
        ]
        self._rel_positions = np.zeros((nb_rays, 2))
        self._distances = np.zeros(nb_rays)
        self._sphere_ids = [
            -1,
        ] * self._nb_rays

        self._observation_limits = np.repeat(
                np.array([[-ray_length], [ray_length]]),
                nb_rays * 2,
                axis=1,
                )

    def get_observation_size(self):
        """Getter for the dimension of the observation space."""
        if self._raw_data:
            return self._nb_rays
        return self._nb_rays * 2

    def get_observation_space(self, obstacles: dict, goals: dict):
        """Create observation space, all observations should be inside the
        observation space."""
        observation_space = gym.spaces.Box(
            -self._ray_length - 0.01,
            self._ray_length + 0.01,
            shape=(self.get_observation_size(),),
            dtype=float,
        )
        return gym.spaces.Dict({self._name: observation_space})



    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        """Sense the distance toward the next object with the Lidar."""
        self._call_counter += 1
        if not self._link_id:
            self._link_id = self._physics_engine.extract_link_id(robot, self._link_name)
        lidar_position = self._physics_engine.get_link_position(robot, self._link_id)
        ray_start = lidar_position
        quat = self._physics_engine.get_link_orientation(robot, self._link_id)
        yaw = Rotation.from_quat(quat).as_euler('zyx', degrees=True)[0]

        for i, theta in enumerate(self._thetas):
            ray_end = np.array(ray_start) + self._ray_length * np.array(
                [np.cos(theta + yaw), np.sin(theta + yaw), 0.0]
            )
            lidar = self._physics_engine.ray_cast(ray_start, ray_end, i, self._ray_length)
            true_rel_positions = (
                lidar
                * np.array([np.cos(theta + yaw), np.sin(theta + yaw)])
            )

            self._rel_positions[i, :] = true_rel_positions
        noisy_rel_positions = self.add_noise(self._rel_positions.flatten()).reshape(self._nb_rays, 2)
        self._rel_positions = noisy_rel_positions

        self._distances = np.linalg.norm(self._rel_positions, axis=1)
        if (
            self._plotting_interval > 0
            and self._call_counter % self._plotting_interval == 0
        ):
            self.update_lidar_spheres(lidar_position)
        if self._raw_data:
            return self._distances
        return self._rel_positions.flatten()

