"""Module for lidar sensor simulation."""
import numpy as np
import pybullet as p
import gym

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
    """

    def __init__(self, linkId, nbRays=10, rayLength=10.0):
        super().__init__("lidarSensor")
        self._nb_rays = nbRays
        self._ray_length = rayLength
        self._link_id = linkId
        self._thetas = [
            i * 2 * np.pi / self._nbRays for i in range(self._nb_rays)
        ]
        self._rel_positions = np.zeros(2 * nbRays)

    def get_observation_size(self):
        """Getter for the dimension of the observation space."""
        return self._nbRays * 2

    def get_observation_space(self):
        """Create observation space, all observations should be inside the
        observation space."""
        return gym.spaces.Box(
            0.0,
            self._rayLength,
            shape=(self.getOSpaceSize(),),
            dtype=np.float64,
        )

    def sense(self, robot):
        """Sense the distance toward the next object with the Lidar."""
        link_state = p.getLinkState(robot, self._linkId)
        lidar_position = link_state[0]
        ray_start = lidar_position
        for i, theta in enumerate(self._thetas):
            ray_end = np.array(ray_start) + self._ray_length * np.array(
                [np.cos(theta), np.sin(theta), 0.0]
            )
            lidar = p.rayTest(ray_start, ray_end)
            self._rel_positions[2 * i : 2 * i + 2] = (
                np.array(lidar[0][3]) - np.array(ray_start)
            )[0:2]
        return self._relPositions
