import numpy as np
import pybullet as p
import gym
import time
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
        for obst in self._obstacles:
            size += 12  # add space for x, xdot, theta and thetadot
        return size

    # def addObstacle(self, obst):
    #     self._obstacles.append(obst)
    #     self._observation = np.zeros(self.getOSpaceSize())

    def getObservationSpace(self):
        # todo: this function
        spacesDict = {}
        for obst in self._obstacles:
            # size = 3 * obst.m() + 1
            spacesDict[obst.name()] = gym.spaces.Dict({
                'pos': gym.spaces.Box(-10, 10, shape=(obst, ), dtype=np.float64),
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
