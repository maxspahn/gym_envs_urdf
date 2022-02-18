import numpy as np
import pybullet as p
import gym

from urdfenvs.sensors.sensor import Sensor


class Lidar(Sensor):

    def __init__(self, linkId, nbRays=10, rayLength=10.0):
        super().__init__("lidarSensor")
        self._nbRays = nbRays
        self._rayLength = rayLength
        self._linkId = linkId
        self._thetas = [i * 2*np.pi/self._nbRays for i in range(self._nbRays)]
        self._relPositions = np.zeros(2*nbRays)

    def getOSpaceSize(self):
        return self._nbRays * 2

    def getObservationSpace(self):
        return gym.spaces.Box(-10, 10, shape=(self.getOSpaceSize(), ), dtype=np.float64)

    def sense(self, robot):
        linkState = p.getLinkState(robot, self._linkId)
        lidarPosition = linkState[0]
        rayStart = lidarPosition
        for i, theta in enumerate(self._thetas):
            rayEnd = np.array(rayStart) + self._rayLength * np.array([np.cos(theta), np.sin(theta), 0.0])
            lidar = p.rayTest(rayStart, rayEnd)
            self._relPositions[2*i:2*i+2] = (np.array(lidar[0][3]) - np.array(rayStart))[0:2]
        return self._relPositions
