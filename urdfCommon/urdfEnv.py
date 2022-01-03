import gym
import time
import numpy as np
import pybullet as p

from abc import abstractmethod
from urdfCommon.plane import Plane


class WrongObservationError(Exception):
    pass


class UrdfEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, robot, render=False, dt=0.01):
        self._dt = dt
        self._t = 0.0
        self.np_random, _ = gym.utils.seeding.np_random()
        self.robot = robot
        self._render = render
        self.clientId = -1
        self.done = False
        self._numSubSteps = 20
        self._nSteps = 0
        self._maxSteps = 10000000
        self._obsts = []
        self._goals = []
        if self._render:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

    def n(self):
        return self.robot.n()

    @abstractmethod
    def setSpaces(self):
        pass

    @abstractmethod
    def applyAction(self, action):
        pass

    def dt(self):
        return self._dt

    def t(self):
        return self._t

    def step(self, action):
        self._t += self.dt()
        # Feed action to the robot and get observation of robot's state
        self._nSteps += 1
        self.applyAction(action)
        for obst in self._obsts:
            obst.updateBulletPosition(p, t=self.t())
        for goal in self._goals:
            goal.updateBulletPosition(p, t=self.t())
        p.stepSimulation()
        ob = self._get_ob()

        # Done by running off boundaries
        reward = 1.0

        if self._nSteps > self._maxSteps:
            reward = reward + 1
            self.done = True
        if self._render:
            self.render()
        return ob, reward, self.done, {}

    def _get_ob(self):
        observation = self.robot.get_observation()
        __import__('pdb').set_trace()
        if not self.observation_space.contains(observation):
            __import__('pdb').set_trace()
            raise WrongObservationError("The observation does not fit the defined observation space")
        return observation

    def addObstacle(self, obst):
        self._obsts.append(obst)
        obst.add2Bullet(p)

    def addGoal(self, goal):
        self._goals.append(goal)
        goal.add2Bullet(p)

    def setWalls(self, limits=[[-2, -2], [2, 2]]):
        colwallId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 10.0, 0.5])
        p.createMultiBody(0, colwallId, 10, [limits[0][0], 0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))
        p.createMultiBody(0, colwallId, 10, [limits[1][0], 0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))
        p.createMultiBody(0, colwallId, 10, [0, limits[0][1], 0.0], p.getQuaternionFromEuler([0, 0, np.pi/2]))
        p.createMultiBody(0, colwallId, 10, [0, limits[1][1], 0.0], p.getQuaternionFromEuler([0, 0, np.pi/2]))

    def addSensor(self, sensor):
        self.robot.addSensor(sensor)
        self.observation_space = gym.spaces.Dict({
            "jointStates": self.observation_space,
            "sensor1": gym.spaces.Box(-10, 10, shape=(sensor.getOSpaceSize(), )),
        })

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def checkInitialState(self, pos, vel):
        if not isinstance(pos, np.ndarray) or not pos.size == self.robot.n():
            pos = np.zeros(self.robot.n())
        if not isinstance(vel, np.ndarray) or not vel.size == self.robot.n():
            vel = np.zeros(self.robot.n())
        return pos, vel

    def reset(self, initialSet=False, pos=None, vel=None):
        self._t = 0.0
        pos, vel = self.checkInitialState(pos, vel)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._numSubSteps
        )
        self.robot.reset(pos=pos, vel=vel)
        self.plane = Plane()
        p.setGravity(0, 0, -10.0)
        p.stepSimulation()
        return self.robot.get_observation()

    def render(self, mode="none"):
        time.sleep(self.dt())
        return

    def close(self):
        p.disconnect()
