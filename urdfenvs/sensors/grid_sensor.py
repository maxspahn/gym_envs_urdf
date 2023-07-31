"""Module for occupancy sensor simulation."""
from abc import abstractmethod
from time import perf_counter
import numpy as np
import gymnasium as gym

from urdfenvs.sensors.sensor import Sensor


class LinkIdNotFoundError(Exception):
    pass


class GridSensor(Sensor):
    def __init__(
        self,
        limits: np.ndarray = np.array([[-1, -1], [-1, 1], [-1, 1]]),
        resolution: np.ndarray = np.array([10, 10, 10], dtype=int),
        interval: int = -1,
        name: str = "Grid",
    ):
        super().__init__(name)
        self._resolution = resolution
        self._limits = limits
        self._interval = interval
        self._call_counter = 13
        self._computed = False
        x_values = np.linspace(limits[0][0], limits[0][1], resolution[0])
        y_values = np.linspace(limits[1][0], limits[1][1], resolution[1])
        z_values = np.linspace(limits[2][0], limits[2][1], resolution[2])
        self._mesh = np.stack(
            np.meshgrid(x_values, y_values, z_values, indexing="ij"), axis=-1
        )
        self._grid_values = np.zeros(shape=self._mesh.shape[0:3], dtype=int)

    def mesh(self) -> np.ndarray:
        return self._mesh

    def get_observation_size(self):
        return self._grid_values.shape

    @abstractmethod
    def get_observation_space(self, obstacles: dict, goals: dict):
        """Create observation space, all observations should be inside the
        observation space."""
        pass

    def distances(self, obstacles: dict) -> np.ndarray:
        mesh_flat = self._mesh.reshape((-1, 3))
        distances = np.min(
            np.array(
                [
                    obstacle.distance(mesh_flat)
                    for obstacle in list(obstacles.values())
                ]
            ),
            axis=0,
        )
        return distances

    @abstractmethod
    def sense(self, robot, obstacles: dict, goals: dict, t: float) -> np.ndarray:
        pass

