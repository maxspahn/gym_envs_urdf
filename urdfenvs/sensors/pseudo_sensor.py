import numpy as np
import pybullet as p
import gym
import sys
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
        min = sys.float_info.min
        max = sys.float_info.max

        for obj_id in range(2, p.getNumBodies()):
            spacesDict[str(obj_id)] = gym.spaces.Dict({
                # "name": gym.spaces.Box(sys.float_info.min, sys.float_info.max, shape=(3, 1), dtype=str),
                "x": gym.spaces.Box(low=min, high=max, shape=(3, 1), dtype=np.float64),
                "xdot": gym.spaces.Box(low=min, high=max, shape=(3, 1), dtype=np.float64),
                "theta": gym.spaces.Box(low=-2*np.pi, high=2*np.pi, shape=(3, 1), dtype=np.float64),
                "thetadot": gym.spaces.Box(low=min, high=max, shape=(3, 1), dtype=np.float64)
             })


        # todo: what is a goal? Should i have a goal observation space? for a sensor?

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

            observation[obj_id] = {"name": p.getBodyInfo(obj_id)[1],
                                    "x": pos[0],
                                    "xdot": vel[0],
                                    "theta": p.getEulerFromQuaternion(pos[1]),
                                    "thetadot": vel[1]
                                   }
        return observation
