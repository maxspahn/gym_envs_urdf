"""Module for occupancy sensor simulation."""
from time import perf_counter
import logging

import numpy as np
import gymnasium as gym

from urdfenvs.sensors.grid_sensor import GridSensor


class OccupancySensor(GridSensor):
    def __init__(
        self,
        limits: np.ndarray = np.array([[-1, -1], [-1, 1], [-1, 1]]),
        resolution: np.ndarray = np.array([10, 10, 10], dtype=int),
        interval: int = -1,
        variance: float = 0.0,
        plotting_interval: int = -1,
        physics_engine_name: str = 'pybullet',
    ):
        super().__init__(
            limits=limits,
            resolution=resolution,
            interval=interval,
            name="Occupancy",
            variance=variance,
            plotting_interval=plotting_interval,
            physics_engine_name=physics_engine_name,
        )
        self._voxel_ids = [
            -1,
        ] * self.number_of_voxels()
        self._voxel_size = self.voxel_size() * 1.0
        self._bullet_ids = []

    def get_observation_space(self, obstacles: dict, goals: dict):
        """Create observation space, all observations should be inside the
        observation space."""
        observation_space = gym.spaces.Box(
            0,
            1,
            shape=self.get_observation_size(),
            dtype=int,
        )
        return gym.spaces.Dict({self._name: observation_space})

    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        self._compute_call_counter += 1
        self._call_counter += 1
        if not (
            self._computed
            and (self._interval < 0 or self._call_counter % self._interval != 0)
        ):
            start_time = perf_counter()
            distances = self.distances(obstacles, t)
            self._grid_values = np.array(distances <= 0.0, dtype=int).reshape(
                self._resolution
            )
            end_time = perf_counter()

            logging.info(f"Computed Occupancy in {end_time-start_time} s")
            self._computed = True
        if (
            self._plotting_interval > 0
            and self._call_counter % self._plotting_interval == 0
        ):
            self.update_occupancy_visualization()
        return self._grid_values
