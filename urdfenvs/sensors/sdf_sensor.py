"""Module for signed distance field sensor simulation."""
from time import perf_counter
import numpy as np
import pybullet as p
import gym

from urdfenvs.sensors.sensor import Sensor

class LinkIdNotFoundError(Exception):
    pass


class SDFSensor(Sensor):
    def __init__(self,
                 limits: np.ndarray =  np.array([[-1, -1], [-1, 1], [-1, 1]]),
                 resolution: np.ndarray = np.array([10, 10, 10], dtype=int),
                 static: bool = True,
                ):
        super().__init__("SDFSensor")
        self._resolution = resolution
        self._limits = limits
        self._static = static
        self._computed = False
        x_values = np.linspace(limits[0][0], limits[0][1], resolution[0])
        y_values = np.linspace(limits[1][0], limits[1][1], resolution[1])
        z_values = np.linspace(limits[2][0], limits[2][1], resolution[2])
        self._mesh = np.stack(np.meshgrid(x_values, y_values, z_values), axis=-1)
        self._sdf = np.zeros(shape=self._mesh.shape[0:3])

    def mesh(self) -> np.ndarray:
        return self._mesh

    def get_observation_size(self):
        return self._sdf.shape

    def get_observation_space(self, obstacles: dict, goals: dict):
        """Create observation space, all observations should be inside the
        observation space."""
        observation_space = gym.spaces.Box(
            0.0,
            10.0,
            shape=self.get_observation_size(),
            dtype=np.float64,
        )
        return gym.spaces.Dict({self._name: observation_space})

    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        if self._static and self._computed:
            return self._sdf
        t0 = perf_counter()
        for i in range(self._resolution[0]):
            for j in range(self._resolution[1]):
                for k in range(self._resolution[2]):
                    pos = self._mesh[j, i, k]
                    distance = np.min(np.array([obstacle.distance(pos) for obstacle in list(obstacles.values())]))
                    self._sdf[j, i, k] = np.maximum(0.0, distance)
        t1 = perf_counter()
        print(f"Computed SDF in {t1-t0}s")
        self._computed = True
        return self._sdf


