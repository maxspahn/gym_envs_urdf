"""Mujoco engine interfaces for sensors."""

from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from urdfenvs.sensors.physics_engine_interface import PhysicsEngineInterface


class MujocoInterface(PhysicsEngineInterface):
    """
    Physics engine interface for mujoco.
    """

    def extract_link_id(self, robot, link_name: str) -> int:
        return self._data.body(link_name).id

    def set_data(self, data) -> None:
        """Set pointer to mujoco data to sensor."""
        self._data = data

    def get_link_position(self, robot, link_id) -> np.ndarray:
        link_position = self._data.xpos[link_id]
        return link_position

    def get_link_orientation(self, robot, link_id) -> np.ndarray:
        link_orientation_matrix = np.reshape(self._data.xmat[link_id], (3, 3))
        return Rotation.from_matrix(link_orientation_matrix).as_quat()

    def get_obstacle_pose(
        self, obst_id: int, obst_name: str, movable: bool = False
    ) -> Tuple[List[float], List[float]]:
        if movable:

            free_joint_data = self._data.jnt(f"freejoint_{obst_name}").qpos
            return free_joint_data[0:3].tolist(), free_joint_data[3:].tolist()
        pos = self._data.body(obst_name).xpos
        ori = self._data.body(obst_name).xquat
        return pos.tolist(), ori.tolist()

    def get_goal_pose(self, goal_id: int) -> Tuple[List[float]]:
        pos = self._data.site(goal_id).xpos
        goal_rotation = np.reshape(self._data.site(goal_id).xmat, (3, 3))
        ori = Rotation.from_matrix(goal_rotation).as_quat()
        return pos.tolist(), ori.tolist()

    def get_goal_velocity(self, goal_id: int) -> Tuple[List[float]]:
        return [0, 0, 0], [0, 0, 0]

    def ray_cast(
        self,
        ray_start: np.ndarray,
        ray_end: np.ndarray,
        ray_index: int,
        ray_length: float,
    ) -> np.ndarray:
        if self._data.sensordata[ray_index] < 0:
            ray_value = ray_length - (0.01 / 2)
        else:
            ray_value = self._data.sensordata[ray_index] - (0.01 / 2)
        return ray_value

    def get_obstacle_velocity(
        self, obst_id: int, obst_name: str, movable: bool = False
    ) -> None:
        raise NotImplementedError("Obstacle velocity not implemented for mujoco.")
