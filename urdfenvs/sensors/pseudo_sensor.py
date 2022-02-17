import numpy as np
import pybullet as p
import gym
from urdfenvs.sensors.sensor import Sensor


class PseudoSensor(Sensor):
    """
    the PseudoSensor class is a sensor sensing the exact position of every object. The PseudoSensor is thus
    a full information sensor which in the real world can never exist. The PseudoSensor returns a dictionary
    with the position of every object when the sense function is called.
    """

    def __init__(self, obstacles=[], goals=[]):
        super().__init__("pseudoSensor")
        self._obstacles = obstacles
        self._observation = np.zeros(self.getOSpaceSize())
        self._goals = goals

    def getOSpaceSize(self):
        size = 0
        for obj_id in range(2, p.getNumBodies()):
            size += 12  # add space for x, xdot, theta and thetadot
        return size

    def getObservationSpace(self):

        spacesDict = gym.spaces.Dict()
        min_os_value = -1000
        max_os_value = 1000

        for obj_id in range(2, p.getNumBodies()):
            spacesDict[str(obj_id)] = gym.spaces.Dict({
                "x": gym.spaces.Box(low=min_os_value, high=max_os_value, shape=(3, ), dtype=np.float64),
                "xdot": gym.spaces.Box(low=min_os_value, high=max_os_value, shape=(3, ), dtype=np.float64),
                "theta": gym.spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(3, ), dtype=np.float64),
                "thetadot": gym.spaces.Box(low=min_os_value, high=max_os_value, shape=(3, ), dtype=np.float64)
             })

        return spacesDict

    def sense(self, robot):
        """
        Sense the exact position of all the objects.
        """
        observation = {}

        # assumption: p.getBodyInfo(0), p.getBodyInfo(1) are the robot and ground plane respectively
        for obj_id in range(2, p.getNumBodies()):

            pos = p.getBasePositionAndOrientation(obj_id)
            vel = p.getBaseVelocity(obj_id)

            observation[str(obj_id)] = {
                "x": pos[0],
                "xdot": vel[0],
                "theta": p.getEulerFromQuaternion(pos[1]),
                "thetadot": vel[1]
                }

        return observation
