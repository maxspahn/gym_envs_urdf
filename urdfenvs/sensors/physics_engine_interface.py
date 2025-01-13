"""Physics engine interfaces for sensors."""
from typing import Tuple, List
from abc import abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation

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


