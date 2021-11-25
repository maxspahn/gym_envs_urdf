import gym
import numpy as np
import time
import pybullet as p

from abc import abstractmethod


class UrdfEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, robot, render=False, dt=0.01):
        self._dt = dt
        self.np_random, _ = gym.utils.seeding.np_random()
        self.robot = robot
        self._render = render
        self.clientId = -1
        self.done = False
        self._numSubSteps = 20
        self._nSteps = 0
        self._maxSteps = 10000000
        if self._render:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

    @abstractmethod
    def setSpaces(self):
        pass

    @abstractmethod
    def applyAction(self, action):
        pass

    def dt(self):
        return self._dt

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

    def addObstacle(self, pos, filename):
        self.robot.addObstacle(pos, filename)

    def addSensor(self, sensor):
        self.robot.addSensor(sensor)
        self.observation_space = gym.spaces.Dict({
            "jointStates": self.observation_space,
            "sensor1": gym.spaces.Box(-10, 10, shape=(sensor.getOSpaceSize(), )),
        })

    def setWalls(self, limits=[[-2, -2], [2, 2]]):
        self.robot.setWalls(limits)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    @abstractmethod
    def reset(self, initialSet=False, pos=np.zeros(2), vel=np.zeros(2)):
        pass

    def render(self, mode="none"):
        time.sleep(self.dt())
        return

    def close(self):
        p.disconnect()
