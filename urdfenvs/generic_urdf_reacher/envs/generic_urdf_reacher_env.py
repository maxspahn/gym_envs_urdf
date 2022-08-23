import numpy as np
from urdfenvs.generic_urdf_reacher.resources.generic_urdf_reacher import GenericUrdfReacher
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class GenericUrdfReacherEnv(UrdfEnv):
    def __init__(self, urdf, **kwargs):
        super().__init__(
            GenericUrdfReacher(urdf), **kwargs
        )

    def check_initial_state(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self._robot.n():
            center_position = (self._robot._limit_pos_j[0] + 
                    self._robot._limit_pos_j[1])/2
            pos = center_position
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel
