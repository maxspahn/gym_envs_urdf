"""Module for signed distance field sensor simulation."""
from time import perf_counter
import numpy as np
import pybullet as p
import gymnasium as gym

from urdfenvs.sensors.sensor import Sensor


class LinkIdNotFoundError(Exception):
    pass


class SDFSensor(Sensor):
    def __init__(
        self,
        limits: np.ndarray = np.array([[-1, -1], [-1, 1], [-1, 1]]),
        resolution: np.ndarray = np.array([10, 10, 10], dtype=int),
        interval: int = -1,
    ):
        super().__init__("SDFSensor")
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
            dtype=float,
        )
        return gym.spaces.Dict({self._name: observation_space})

    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        self._call_counter += 1
        if self._computed and (
            self._interval < 0 or self._call_counter % self._interval != 0
        ):
            return self._sdf
        start_time = perf_counter()
        mesh_flat = self._mesh.reshape((-1, 3))
        self._sdf = np.maximum(
            np.min(
                np.array(
                    [
                        obstacle.distance(mesh_flat)
                        for obstacle in list(obstacles.values())
                    ]
                ),
                axis=0,
            ),
            0.0,
        ).reshape(self._resolution)
        end_time = perf_counter()

        print(f"Computed SDF in {end_time-start_time} s")
        self._computed = True
        return self._sdf
