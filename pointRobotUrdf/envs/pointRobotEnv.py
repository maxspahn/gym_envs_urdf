import numpy as np
import pybullet as p
from abc import abstractmethod

from pointRobotUrdf.resources.pointRobot import PointRobot
from pointRobotUrdf.resources.plane import Plane
from urdfCommon.urdfEnv import UrdfEnv


class PointRobotEnv(UrdfEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False, dt=0.01):
        super().__init__(PointRobot(), render=render, dt=dt)
        self.setSpaces()
        self.reset(initialSet=True)

    @abstractmethod
    def setSpaces(self):
        pass

    @abstractmethod
    def applyAction(self, action):
        pass

    def reset(self, initialSet=False, pos=np.zeros(2), vel=np.zeros(2)):
        if not initialSet:
            print("Run " + str(self._nSteps) + " steps in this run")
            self._nSteps = 0
            p.resetSimulation()
        p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._numSubSteps
        )
        self.plane = Plane()
        self.robot.reset(pos=pos, vel=vel)
        p.setGravity(0, 0, -10)
        p.stepSimulation()

        # Get observation to return
        return self.robot.get_observation()
