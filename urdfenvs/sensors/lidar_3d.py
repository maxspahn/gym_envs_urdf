"""Module for lidar sensor simulation."""
from typing import List, Optional
import numpy as np
import pybullet as p
import gymnasium as gym

from urdfenvs.sensors.sensor import Sensor


class LinkIdNotFoundError(Exception):
    pass


class Lidar3D(Sensor):
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
        nb_rays: Optional[List[int]] = None,
        ray_length=10.0,
        raw_data=True,
        angle_limits: np.ndarray = np.array(
            [[-np.pi, np.pi], [-0.1 * np.pi, 0.1 * np.pi]]
        ),
        variance: float = 0.0,
    ):
        super().__init__("LidarSensor", variance=variance)
        if nb_rays is None:
            self._nb_rays = [10, 10]
        else:
            self._nb_rays = nb_rays
        self._total_rays = nb_rays[0] * nb_rays[1]
        self._raw_data = raw_data
        self._ray_length = ray_length
        self._link_name = link_name
        self._link_id = None
        if isinstance(link_name, int):
            self._link_id = link_name
        self._angle_limits = angle_limits
        self._thetas = [
            angle_limits[0, 0]
            + i * (angle_limits[0, 1] - angle_limits[0, 0]) / self._nb_rays[0]
            for i in range(self._nb_rays[0])
        ]
        self._alphas = [
            angle_limits[1, 0]
            + i * (angle_limits[1, 1] - angle_limits[1, 0]) / self._nb_rays[1]
            for i in range(self._nb_rays[1])
        ]
        self._rel_positions = np.zeros(3 * self._total_rays)
        self._distances = np.zeros(self._total_rays)
        self._sphere_ids = [
            -1,
        ] * self._total_rays

    def get_observation_size(self):
        """Getter for the dimension of the observation space."""
        if self._raw_data:
            return self._nb_rays
        return self._total_rays * 3

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

    def extract_link_id(self, robot):
        number_links = p.getNumJoints(robot)
        joint_names = []
        for i in range(number_links):
            joint_name = p.getJointInfo(robot, i)[1].decode("UTF-8")
            joint_names.append(joint_name)
            if joint_name == self._link_name:
                self._link_id = i
                return
        raise LinkIdNotFoundError(
            f"Link with name {self._link_name} not found. "
            f"Possible links are {joint_names}"
        )

    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        """Sense the distance toward the next object with the Lidar."""
        if not self._link_id:
            self.extract_link_id(robot)
        link_state = p.getLinkState(robot, self._link_id)

        lidar_position = link_state[0]
        ray_start = (lidar_position[0], lidar_position[1], lidar_position[2])
        yaw = p.getEulerFromQuaternion(link_state[1])[2]
        for i, theta in enumerate(self._thetas):
            for j, alpha in enumerate(self._alphas):
                direction = np.array(
                    [
                        np.cos(alpha) * np.cos(theta + yaw),
                        np.cos(alpha) * np.sin(theta + yaw),
                        np.sin(alpha),
                    ]
                )
                ray_end = np.array(ray_start) + self._ray_length * direction
                lidar = p.rayTest(ray_start, ray_end)
                index = 3 * self._nb_rays[1] * i + 3 * j
                actual_ray_end = lidar[0][2] * self._ray_length * direction
                self._rel_positions[index : index + 3] = actual_ray_end
                self._distances[i * self._nb_rays[1] + j] = np.linalg.norm(
                    self._rel_positions[index : index + 3]
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
        q_obs = self._rel_positions.reshape(self._total_rays, 3)
        shape_id_sphere = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.0, 0.0, 0.0, 0.8]
        )
        for ray_id in range(self._total_rays):
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
        q_obs = self._rel_positions.reshape(self._total_rays, 3)
        for ray_id in range(self._total_rays):
            p.resetBasePositionAndOrientation(
                int(self._sphere_ids[ray_id]), q + q_obs[ray_id], [0, 0, 0, 1]
            )
