import numpy as np

from urdfenvs.albert_reacher.resources.albert_robot import AlbertRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class AlbertReacherEnv(UrdfEnv):
    """Albert reacher environment."""

    def __init__(self, render=False, dt=0.01):
        super().__init__(AlbertRobot(), render=render, dt=dt)
        self.set_spaces()

    def check_initial_state(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self._robot.n()+1:
            pos = np.zeros(self._robot.n()+1)
            pos[6] = -1.501
            pos[8] = 1.8675
            pos[9] = np.pi/4
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel
