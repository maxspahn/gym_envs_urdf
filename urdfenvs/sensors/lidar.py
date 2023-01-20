"""Module for lidar sensor simulation."""
import numpy as np
import pybullet as p
import gym
from typing import List

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

    def __init__(self, link_id, nb_rays=10, ray_length=10.0, raw_data=True):
        super().__init__("LidarSensor")
        self._nb_rays = nb_rays
        self._raw_data = raw_data
        self._ray_length = ray_length
        self._link_id = link_id
        self._thetas = [
            i * 2 * np.pi / self._nb_rays for i in range(self._nb_rays)
        ]
        self._rel_positions = np.zeros(2 * nb_rays)
        self._distances = np.zeros(nb_rays)
        self._sphere_ids = [
            -1,
        ] * self._nb_rays

    def get_observation_size(self):
        """Getter for the dimension of the observation space."""
        if self._raw_data:
            return self._nb_rays
        return self._nb_rays * 2

    def get_observation_space(self):
        """Create observation space, all observations should be inside the
        observation space."""
        return gym.spaces.Box(
            -self._ray_length,
            self._ray_length,
            shape=(self.get_observation_size(),),
            dtype=np.float64,
        )

    def sense(self, robot, obst_ids: List[int], goal_ids: List[int]):
        """Sense the distance toward the next object with the Lidar."""
        link_state = p.getLinkState(robot, self._link_id)
        lidar_position = link_state[0]
        ray_start = lidar_position
        # Alpha and gamma both are the angle of the robot rotated along the z-axis.
        alpha = np.arcsin(link_state[1][2]) * 2
        gamma = np.arccos(link_state[1][3]) * 2
        for i, theta in enumerate(self._thetas):
            ray_end = np.array(ray_start) + self._ray_length * np.array(
                [np.cos(theta + alpha), np.sin(theta + gamma), 0.0]
            )
            lidar = p.rayTest(ray_start, ray_end)
            self._rel_positions[2 * i : 2 * i + 2] = (
                lidar[0][2]
                * self._ray_length
                * np.array([np.cos(theta + alpha), np.sin(theta + gamma)])
            )
            self._distances[i] = np.linalg.norm(
                self._rel_positions[2 * i : 2 * i + 2]
            )
        self.update_lidar_spheres(lidar_position)
        if self._raw_data:
            return self._distances
        return self._rel_positions

    def init_lidar_spheres(self, lidar_position):
        """
        Initialize the Lidar spheres to visualize the sensing in bullet.
        The visual spheres are initialized once and then update in the
        update function. The relative positions are augmented by a third
        column to have a full position, including the z-coordinate.

        Parameters
        ------------
        lidar_position : np.ndarray
            The position of the lidar sensor link.
        """
        q = lidar_position
        q_obs = self._rel_positions.reshape(self._nb_rays, 2)
        q_obs = np.append(q_obs, np.zeros((self._nb_rays, 1)), axis=1)
        shape_id_sphere = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.0, 0.0, 0.0, 0.8]
        )
        for ray_id in range(self._nb_rays):
            body_id_sphere = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=shape_id_sphere,
                basePosition=q + q_obs[ray_id],
            )
            self._sphere_ids[ray_id] = body_id_sphere

    def update_lidar_spheres(self, lidar_position):
        """
        Updates the position of the spheres visualizing the sensing with lidar.
        If the spheres have not been initialized, the init_lidar_spheres
        function is called.

        Parameters
        ------------
        lidar_position : np.ndarray
            The position of the lidar sensor link.
        """
        if self._sphere_ids[0] == -1:
            self.init_lidar_spheres(lidar_position)
        q = lidar_position
        # Reshape and add z-values to the sensor data.
        q_obs = self._rel_positions.reshape(self._nb_rays, 2)
        q_obs = np.append(q_obs, np.zeros((self._nb_rays, 1)), axis=1)
        for ray_id in range(self._nb_rays):
            p.resetBasePositionAndOrientation(
                int(self._sphere_ids[ray_id]), q + q_obs[ray_id], [0, 0, 0, 1]
            )
