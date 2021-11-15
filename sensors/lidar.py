import numpy as np
import pybullet as p

class Lidar(object):

    def __init__(self, linkId, nbRays=10, rayLength=10.0):
        self._nbRays = nbRays
        self._rayLength = rayLength
        self._linkId = linkId
        self._thetas = [i * 2*np.pi/self._nbRays for i in range(self._nbRays)]
        self._relPositions = np.zeros(2*nbRays)

    def getOSpaceSize(self):
        return self._nbRays * 2

    def sense(self, robot):
        linkState = p.getLinkState(robot, self._linkId)
        lidarPosition = linkState[0]
        rayStart = lidarPosition
        thetas = []
        for i, theta in enumerate(self._thetas):
            rayEnd = np.array(rayStart) + self._rayLength * np.array([np.cos(theta), np.sin(theta), 0.0])
            lidar = p.rayTest(rayStart, rayEnd)
            self._relPositions[2*i:2*i+2] = (np.array(lidar[0][3]) - np.array(rayStart))[0:2]
        return self._relPositions
