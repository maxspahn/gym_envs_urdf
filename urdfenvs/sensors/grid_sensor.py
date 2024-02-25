"""Module for occupancy sensor simulation."""
from abc import abstractmethod
from typing import Tuple

import numpy as np

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
        variance: float = 0.0,
        plotting_interval: int = -1,
        physics_engine_name: str = "pybullet",
    ):
        super().__init__(
            name,
            variance=variance,
            plotting_interval=plotting_interval,
            physics_engine_name=physics_engine_name,
        )
        self._resolution = resolution
        self._limits = limits
        self._interval = interval
        self._compute_call_counter = 13
        self._computed = False
        x_values = np.linspace(limits[0][0], limits[0][1], resolution[0])
        y_values = np.linspace(limits[1][0], limits[1][1], resolution[1])
        z_values = np.linspace(limits[2][0], limits[2][1], resolution[2])
        self._mesh = np.stack(
            np.meshgrid(x_values, y_values, z_values, indexing="ij"), axis=-1
        )
        self._mesh_flat = self.mesh().reshape((-1, 3))
        self._grid_values = np.zeros(shape=self._mesh.shape[0:3], dtype=int)

    def mesh(self) -> np.ndarray:
        return self._mesh

    def number_of_voxels(self) -> int:
        return self._mesh.size // 3

    def voxel_size(self) -> Tuple[float]:
        voxel_size = (self._limits[:, 1] - self._limits[:, 0]) / (
            self._resolution - 1
        )
        nan_index = np.where(np.isnan(voxel_size))[0]
        voxel_size[nan_index] = np.maximum(
            0.01, self._limits[nan_index, 1] - self._limits[nan_index, 0]
        )
        return voxel_size

    def get_observation_size(self):
        return self._grid_values.shape

    @abstractmethod
    def get_observation_space(self, obstacles: dict, goals: dict):
        """Create observation space, all observations should be inside the
        observation space."""
        pass

    def distances(self, obstacles: dict, t: float) -> np.ndarray:
        max_voxel_size = np.max(self.voxel_size()) / 2.0
        mesh_flat = self._mesh.reshape((-1, 3))
        if obstacles:
            distances = np.min(
                np.array(
                    [
                        obstacle.distance(mesh_flat, t=t) - max_voxel_size
                        for obstacle in list(obstacles.values())
                    ]
                ),
                axis=0,
            )
            noisy_distances = np.random.normal(distances, self._variance)
            return noisy_distances
        else:
            return np.ones(self.number_of_voxels()) * 1000

    @abstractmethod
    def sense(
        self, robot, obstacles: dict, goals: dict, t: float
    ) -> np.ndarray:
        pass
