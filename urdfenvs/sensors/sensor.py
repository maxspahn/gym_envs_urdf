"""Abstract class for sensor."""
from typing import List
from abc import abstractmethod


class Sensor(object):
    """Abstract sensor class.

    This class serves as a blueprint for sensors. Every sensor must
    inherit from this class and implement the abstract methods.
    """

    def __init__(
        self, name, variance: float = 0.0, plotting_interval: int = -1
    ):
        self._name = name
        self._variance = variance
        self._plotting_interval = plotting_interval
        self._call_counter = plotting_interval - 1

    def name(self):
        return self._name

    @abstractmethod
    def get_observation_size(self):
        pass

    @abstractmethod
    def get_observation_space(self, obstacles: dict, goals: dict):
        pass

    @abstractmethod
    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        pass
