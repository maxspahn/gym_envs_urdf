import numpy as np

from urdfenvs.iris.resources.iris import IRIS
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class IRISEnv(UrdfEnv):
    def __init__(self, **kwargs):
        super().__init__(IRIS(), **kwargs)

    def check_initial_state(self, pos, vel):
        if (
            not isinstance(pos, np.ndarray)
            or not pos.size == self._robot.n() + 3
        ):
            pos = np.zeros(self._robot.n() + 3)
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel
