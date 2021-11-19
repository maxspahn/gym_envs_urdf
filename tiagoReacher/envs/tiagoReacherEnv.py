import gym
import numpy as np
from abc import abstractmethod
import time
import pybullet as p
from pybullet_utils import bullet_client
from tiagoReacher.resources.tiagoRobot import TiagoRobot
from albertReacher.resources.plane import Plane


class TiagoReacherEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False, dt=0.01):
        self._n = 19
        self._dt = dt
        self.np_random, _ = gym.utils.seeding.np_random()
        self.robot = TiagoRobot()
        (self.observation_space, self.action_space) = self.robot.getVelSpaces()
        self._render = render
        self.done = False
        self._numSubSteps= 2
        self._nSteps = 0
        self._maxSteps = 10000000
        if self._render:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

    def dt(self):
        return self._dt

    def n(self):
        return self._n

    def step(self, action):
        # Feed action to the robot and get observation of robot's state
        self._nSteps += 1
        self.applyAction(action)
        p.stepSimulation()
        ob = self.robot.get_observation()

        # Done by running off boundaries
        reward = 1.0
        if self._nSteps > self._maxSteps:
            reward = reward + 1
            self.done = True
        if self._render:
            self.render()
        return ob, reward, self.done, {}

    @abstractmethod
    def applyAction(self, action):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, initialSet=False):
        if not initialSet:
            print("Run " + str(self._nSteps) + " steps in this run")
            self._nSteps = 0
            p.resetSimulation()
        p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._numSubSteps
        )
        self.plane = Plane()
        self.robot.reset()
        p.setGravity(0, 0, -10)
        p.stepSimulation()
        return self.robot.get_observation()

    def render(self, mode="none"):
        time.sleep(self.dt())
        return

    def close(self):
        p.disconnect()
