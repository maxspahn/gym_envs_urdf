import gym
import numpy as np
import time
import pybullet as p
from pybullet_utils import bullet_client
from pointRobotUrdf.resources.pointRobot import PointRobot
from pointRobotUrdf.resources.plane import Plane
from pointRobotUrdf.envs.genericEnv import PointRobotEnv


class PointRobotAccEnv(PointRobotEnv):

    def reset(self, initialSet=False, pos=np.zeros(2), vel=np.zeros(2)):
        super().reset(initialSet=initialSet, pos=pos, vel=vel)
        self.robot.disableVelocityControl()

    def step(self, action):
        # Feed action to the robot and get observation of robot's state
        self._nSteps += 1
        self.robot.apply_acc_action(action)
        self._p.stepSimulation()
        ob = self.robot.get_observation()

        # Done by running off boundaries
        reward = 1.0

        if self._nSteps > self._maxSteps:
            reward = reward + 1
            self.done = True
        if self._render:
            self.render()
        return ob, reward, self.done, {}

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getAccSpaces()

