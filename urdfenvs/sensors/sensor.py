"""Abstract class for sensor."""
from abc import abstractmethod
from typing import Optional

import numpy as np

from urdfenvs.sensors.physics_engine_interface import PhysicsEngineInterface


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
            from urdfenvs.sensors.pybullet_interface import PybulletInterface
            self._physics_engine = PybulletInterface()
        elif physics_engine_name == 'mujoco':
            from urdfenvs.sensors.mujoco_interface import MujocoInterface
            self._physics_engine = MujocoInterface()

    def name(self):
        return self._name

    def set_data(self, data):
        self._physics_engine.set_data(data)

    def add_noise(self, exact_data: np.ndarray, limits: Optional[np.ndarray] = None):
        """Add noise to the exact data."""
        noisy_data = np.random.normal(exact_data, self._variance)
        used_limits = limits if limits is not None else self._observation_limits
        
        if np.all(exact_data >= used_limits[0]) and np.all(exact_data <= used_limits[1]):
            clipped = np.clip(noisy_data, used_limits[0], used_limits[1])
            return clipped
        else:
            return noisy_data


    @abstractmethod
    def get_observation_size(self):
        pass

    @abstractmethod
    def get_observation_space(self, obstacles: dict, goals: dict):
        pass

    @abstractmethod
    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        pass
