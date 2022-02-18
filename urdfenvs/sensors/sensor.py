import numpy as np
import pybullet as p
import gym
from abc import abstractmethod


class Sensor(object):

    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    @abstractmethod
    def getOSpaceSize(self):
        pass

    @abstractmethod
    def getObservationSpace(self):
        pass

    @abstractmethod
    def sense(self, robot):
        pass
