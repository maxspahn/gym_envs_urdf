from abc import abstractmethod
import numpy as np

from urdfenvs.mobile_reacher.resources.mobile_robot import MobileRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class MobileReacherEnv(UrdfEnv):
    def __init__(self, render=False, dt=0.01, gripper=False):
        super().__init__(MobileRobot(gripper=gripper), render=render, dt=dt)
        self._n = self._robot.n()
        self.set_spaces()

    def check_initial_state(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self._robot.n():
            pos = np.zeros(self._robot.n())
            pos[6] = -1.501
            pos[8] = 1.8675
            pos[9] = np.pi / 4
            if self._robot.n() > 10:
                pos[10] = 0.02
                pos[11] = 0.02
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel

    @abstractmethod
    def set_spaces(self):
        pass

    @abstractmethod
    def apply_action(self, action):
        pass
