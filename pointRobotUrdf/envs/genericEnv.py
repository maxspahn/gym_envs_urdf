import gym
from gym.spaces import Dict, Box
import numpy as np
import time
import pybullet as p
from pybullet_utils import bullet_client
from pointRobotUrdf.resources.pointRobot import PointRobot
from pointRobotUrdf.resources.plane import Plane

from abc import abstractmethod


class PointRobotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False, dt=0.01):
        self._dt = dt
        self.np_random, _ = gym.utils.seeding.np_random()
        self.robot = PointRobot()
        self.setSpaces()
        self._render = render
        self.clientId = -1
        self.done = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self._numSubSteps= 20
        self._nSteps = 0
        self._maxSteps = 10000000
        self._p = p
        if self._render:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.reset(initialSet=True)

    def addSensor(self, sensor):
        self.robot.addSensor(sensor)
        self.observation_space = Dict({
            "jointStates": self.observation_space,
            "sensor1": Box(-10, 10, shape=(sensor.getOSpaceSize(), )),
        })

    @abstractmethod
    def setSpaces(self):
        pass

    def dt(self):
        return self._dt

    def setWalls(self, limits=[[-2, -2], [2, 2]]):
        self.robot.setWalls(limits)

    @abstractmethod
    def step(self, action):
        pass

    def addObstacle(self, pos, filename):
        self.robot.addObstacle(pos, filename)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def initSim(self, numSubSteps):
        if self._render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()
        self.clientId = self._p._client

        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=numSubSteps
        )
        self._p.setGravity(0, 0, -10)
        # Load the plane and robot
        self.plane = Plane(self.clientId)
        self.done = False

        # Visual element of the goal
        self.initState = self._p.saveState()

    def reset(self, initialSet=False, pos=np.zeros(2), vel=np.zeros(2)):
        if not initialSet:
            print("Run " + str(self._nSteps) + " steps in this run")
            self._nSteps = 0
            #self._p.restoreState(self.initState)
            p.resetSimulation()
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._numSubSteps
        )
        self.plane = Plane()
        self.robot.reset(pos=pos, vel=vel)
        self._p.setGravity(0, 0, -10)

        p.stepSimulation()

        # Get observation to return
        robot_ob = self.robot.get_observation()

        return robot_ob

    def render(self, mode="none"):
        time.sleep(self.dt())
        return

    def close(self):
        self._p.disconnect()
