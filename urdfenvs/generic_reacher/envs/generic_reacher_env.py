import numpy as np
from urdfenvs.generic_reacher.resources.generic_robot import GenericRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class GenericReacherEnv(UrdfEnv):
    def __init__(self, urdf, **kwargs):
        super().__init__(
            GenericRobot(urdf), **kwargs
        )

    def check_initial_state(self, pos, vel):
        print(self._robot._limit_pos_j)
        if not isinstance(pos, np.ndarray) or not pos.size == self._robot.n():
            center_position = (self._robot._limit_pos_j[0] + self._robot._limit_pos_j[1])/2
            pos = center_position
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel
