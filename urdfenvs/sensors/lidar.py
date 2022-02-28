import numpy as np
import pybullet as p
import gym

from urdfenvs.sensors.sensor import Sensor


class Lidar(Sensor):
    """
    The Lidar sensor senses the distance toward the next object. A maximum sensing distance and the number of rays can
    be set. The Rays are evenly distributed in a circle.
    """

    def __init__(self, linkId, nbRays=10, rayLength=10.0):
        super().__init__("lidarSensor")
        self._nbRays = nbRays
        self._rayLength = rayLength
        self._linkId = linkId
        self._thetas = [i * 2*np.pi/self._nbRays for i in range(self._nbRays)]
        self._relPositions = np.zeros(2*nbRays)

    def getOSpaceSize(self):
        """
        Getter for the dimension of the observation space.
        """
        return self._nbRays * 2

    def getObservationSpace(self):
        """
        Create observation space, all observations should be inside the observation space.
        """
        return gym.spaces.Box(0.0, self._rayLength, shape=(self.getOSpaceSize(), ), dtype=np.float64)

    def sense(self, robot):
        """
        Sense the distance toward the next object with the Lidar.
        """
        linkState = p.getLinkState(robot, self._linkId)
        lidarPosition = linkState[0]
        rayStart = lidarPosition
        for i, theta in enumerate(self._thetas):
            rayEnd = np.array(rayStart) + self._rayLength * np.array([np.cos(theta), np.sin(theta), 0.0])
            lidar = p.rayTest(rayStart, rayEnd)
            self._relPositions[2*i:2*i+2] = (np.array(lidar[0][3]) - np.array(rayStart))[0:2]
        return self._relPositions
