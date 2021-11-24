import numpy as np
from pointRobotUrdf.envs.pointRobotEnv import PointRobotEnv


class PointRobotAccEnv(PointRobotEnv):

    def reset(self, initialSet=False, pos=np.zeros(2), vel=np.zeros(2)):
        ob = super().reset(initialSet=initialSet, pos=pos, vel=vel)
        self.robot.disableVelocityControl()
        return ob

    def applyAction(self, action):
        self.robot.apply_acc_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getAccSpaces()

