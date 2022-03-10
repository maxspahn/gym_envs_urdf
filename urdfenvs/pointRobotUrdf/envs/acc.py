import numpy as np
from urdfenvs.pointRobotUrdf.envs.pointRobotEnv import PointRobotEnv


class PointRobotAccEnv(PointRobotEnv):

    def reset(self, pos=np.zeros(2), vel=np.zeros(2)):
        ob = super().reset(pos=pos, vel=vel)
        self.robot.disableVelocityControl()
        return ob

    def applyAction(self, action):
        self.robot.apply_acceleration_action(action, self.dt())

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getAccelerationSpaces()

