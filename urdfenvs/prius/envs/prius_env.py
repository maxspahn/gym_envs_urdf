import numpy as np

from urdfenvs.prius.resources.prius import Prius
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class PriusEnv(UrdfEnv):
    def __init__(self, **kwargs):
        super().__init__(Prius(), **kwargs)
        self.set_spaces()

    def check_initial_state(self, pos, vel):
        if (
            not isinstance(pos, np.ndarray)
            or not pos.size == self._robot.n() + 1
        ):
            pos = np.zeros(self._robot.n() + 1)
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel
