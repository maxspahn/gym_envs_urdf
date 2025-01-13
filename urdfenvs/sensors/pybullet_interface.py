"""Pybullet engine interfaces for sensors."""

from typing import List, Tuple

import numpy as np
import pybullet
from scipy.spatial.transform import Rotation

from urdfenvs.sensors.physics_engine_interface import PhysicsEngineInterface


class PybulletInterface(PhysicsEngineInterface):
    """
    Physics engine interface for bullet.
    """

    def extract_link_id(self, robot, link_name: str) -> int:
        number_links = pybullet.getNumJoints(robot)
        joint_names = []
        for i in range(number_links):
            joint_name = pybullet.getJointInfo(robot, i)[1].decode("UTF-8")
            joint_names.append(joint_name)
            if joint_name == link_name:
                return i
        raise LinkIdNotFoundError(
            f"Link with name {link_name} not found. "
            f"Possible links are {joint_names}"
        )

    def get_obstacle_pose(
        self, obst_id: int, obst_name: str, movable: bool = False
    ) -> None:
        position, orientation = pybullet.getBasePositionAndOrientation(obst_id)
        return position, orientation

    def get_obstacle_velocity(
        self, obst_id: int, obst_name: str, movable: bool = False
    ) -> None:
        linear, angular = pybullet.getBaseVelocity(obst_id)
        return linear, angular

    def get_goal_pose(self, goal_id: int) -> Tuple[List[float]]:
        position, orientation = pybullet.getBasePositionAndOrientation(goal_id)
        return position, orientation

    def get_goal_velocity(self, goal_id: int) -> Tuple[List[float]]:
        linear, angular = pybullet.getBaseVelocity(goal_id)
        return linear, angular

    def get_link_position(self, robot, link_id) -> np.ndarray:
        link_position = np.array(pybullet.getLinkState(robot, link_id)[0])
        return link_position

    def get_link_orientation(self, robot, link_id) -> np.ndarray:
        link_orientation = np.array(pybullet.getLinkState(robot, link_id)[1])
        return link_orientation

    def ray_cast(
        self,
        ray_start: np.ndarray,
        ray_end: np.ndarray,
        ray_index: int,
        ray_length: float,
    ) -> np.ndarray:
        lidar = pybullet.rayTest(ray_start, ray_end)
        return lidar[0][2] * ray_length

    def clear_visualizations(self) -> None:
        pybullet.removeAllUserDebugItems()

    def add_visualization_line(
        self, start_point: Tuple[float], end_point: Tuple[float]
    ) -> None:
        pybullet.addUserDebugLine(start_point, end_point)
