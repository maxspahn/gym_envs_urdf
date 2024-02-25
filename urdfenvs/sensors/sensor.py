"""Abstract class for sensor."""
from typing import List
from abc import abstractmethod

from urdfenvs.sensors.physics_engine_interface import PhysicsEngineInterface, PybulletInterface, MujocoInterface

class Sensor():
    """Abstract sensor class.

    This class serves as a blueprint for sensors. Every sensor must
    inherit from this class and implement the abstract methods.
    """

    _name: str
    _variance: float
    _plotting_interval: int
    _physics_engine: PhysicsEngineInterface

    def __init__(
        self,
        name: str,
        variance: float = 0.0,
        plotting_interval: int = -1,
        physics_engine_name: str = 'pybullet',
    ):
        self._name = name
        self._variance = variance
        self._plotting_interval = plotting_interval
        self._call_counter = plotting_interval - 1
        if physics_engine_name == 'pybullet':
            self._physics_engine = PybulletInterface()
        elif physics_engine_name == 'mujoco':
            self._physics_engine = MujocoInterface()

    def name(self):
        return self._name

    def set_data(self, data):
        self._physics_engine.set_data(data)

    @abstractmethod
    def get_observation_size(self):
        pass

    @abstractmethod
    def get_observation_space(self, obstacles: dict, goals: dict):
        pass

    @abstractmethod
    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        pass
