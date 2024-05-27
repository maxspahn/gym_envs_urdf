"""Physics engine interfaces for sensors."""
from typing import Tuple, List
from abc import abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation
import pybullet

class LinkIdNotFoundError(Exception):
    pass


class PhysicsEngineInterface():
    """Physics engine interface for sensors.

    This abstract class defines interfaces that a physics engine must define.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_obstacle_pose(self, *args, **kwargs) -> Tuple[List[float], List[float]]:
        pass

    @abstractmethod
    def get_obstacle_velocity(self, *args, **kwargs) -> Tuple[List[float], List[float]]:
        pass

    @abstractmethod
    def get_goal_pose(self, goal_id: int) -> Tuple[List[float]]:
        pass

    @abstractmethod
    def get_goal_velocity(self, goal_id: int) -> Tuple[List[float]]:
        pass

    @abstractmethod
    def get_link_position(self, *args) -> np.ndarray:
        pass

    @abstractmethod
    def get_link_orientation(self, *args) -> np.ndarray:
        pass

    def ray_cast(
        self,
        ray_start: np.ndarray,
        ray_end: np.ndarray,
        ray_index: int,
        ray_length: float
    ) -> np.ndarray:
        raise NotImplementedError(
            f"Clearing of visualization of lines not implemented for {type(self)}."
        )

    def clear_visualizations(self) -> None:
        raise NotImplementedError(
            f"Clearing of visualization of lines not implemented for {type(self)}."
        )

    def add_visualization_line(self, *args) -> None:
        raise NotImplementedError(
            f"Visualization of lines not implemented for {type(self)}."
        )


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


    def get_obstacle_pose(self, obst_id: int, obst_name: str, movable: bool = False) -> None:
        position, orientation = pybullet.getBasePositionAndOrientation(obst_id)
        return position, orientation

    def get_obstacle_velocity(self, obst_id: int, obst_name: str, movable: bool = False) -> None:
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
        ray_length: float
    ) -> np.ndarray:
        lidar = pybullet.rayTest(ray_start, ray_end)
        return lidar[0][2] * ray_length

    def clear_visualizations(self) -> None:
        pybullet.removeAllUserDebugItems()

    def add_visualization_line(
        self, start_point: Tuple[float], end_point: Tuple[float]
    ) -> None:
        pybullet.addUserDebugLine(start_point, end_point)

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

    def get_obstacle_pose(self, obst_id: int, obst_name: str, movable: bool = False) -> Tuple[List[float], List[float]]:
        if movable:

            free_joint_data = self._data.jnt(f"freejoint_{obst_name}").qpos
            print(free_joint_data)
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
        ray_length: float
    ) -> np.ndarray:
        if self._data.sensordata[ray_index] < 0:
            ray_value = ray_length - (0.01 / 2)
        else:
            ray_value = self._data.sensordata[ray_index] - (0.01 / 2)
        return ray_value

    def get_obstacle_velocity(self, obst_id: int, obst_name: str, movable: bool = False) -> None:
        raise NotImplementedError(
            "Obstacle velocity not implemented for mujoco."
        )
