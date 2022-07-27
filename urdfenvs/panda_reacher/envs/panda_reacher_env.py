import numpy as np
from urdfenvs.panda_reacher.resources.panda_robot import PandaRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class PandaReacherEnv(UrdfEnv):
    def __init__(self, friction=0.0, gripper=False, **kwargs):
        super().__init__(
            PandaRobot(gripper=gripper, friction=friction), **kwargs
        )

    def check_initial_state(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self._robot.n():
            pos = np.zeros(self._robot.n())
            pos[3] = -1.501
            pos[5] = 1.8675
            pos[6] = np.pi / 4
            if self._robot.n() > 7:
                pos[7] = 0.02
                pos[8] = 0.02
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel
