import numpy as np
import pybullet as p
import gym

from urdfenvs.sensors.sensor import Sensor


class PseudoSensor(Sensor):

    def __init__(self, obstacles=[], goals=[]):
        super().__init__("pseudoSensor")
        self._obstacles = obstacles
        self._goals = goals
        self._observation = np.zeros(self.getOSpaceSize())

    def getOSpaceSize(self):
        size = 0
        for obst in self._obstacles:
            size += obst.m() * 3 + 1
        for goal in self._goals:
            size += goal.dim() * 3 + 1
        return size

    def getObservationSpace(self):
        spacesDict = {}
        for obst in self._obstacles:
            size = 3 * obst.m() + 1
            spacesDict[obst.name()] = gym.spaces.Dict({
                'pos': gym.spaces.Box(-10, 10, shape=(obst.m(), ), dtype=np.float64), 
                'vel': gym.spaces.Box(-10, 10, shape=(obst.m(), ), dtype=np.float64), 
                'acc': gym.spaces.Box(-10, 10, shape=(obst.m(), ), dtype=np.float64), 
                'r': gym.spaces.Box(-10, 10, shape=(1, ), dtype=np.float64)
            })

        for goal in self._goals:
            size = 3 * goal.dim() + 1
            spacesDict[goal.name()] = gym.spaces.Box(-10, 10, shape=(size, ), dtype=np.float64)
        space = gym.spaces.Dict(spacesDict)
        return space

    def sense(self, robot):
        observation = {}
        for obst in self._obstacles:
            observation[obst.name()] = np.concatenate(obst.position(), obst.velocity(), obst.acceleration()

        
        linkState = p.getLinkState(robot, self._linkId)
        lidarPosition = linkState[0]
        rayStart = lidarPosition
        for i, theta in enumerate(self._thetas):
            rayEnd = np.array(rayStart) + self._rayLength * np.array([np.cos(theta), np.sin(theta), 0.0])
            lidar = p.rayTest(rayStart, rayEnd)
            self._relPositions[2*i:2*i+2] = (np.array(lidar[0][3]) - np.array(rayStart))[0:2]
        return self._relPositions
