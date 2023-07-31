"""Module for occupancy sensor simulation."""
from time import perf_counter
import numpy as np
import gymnasium as gym

from urdfenvs.sensors.grid_sensor import GridSensor


class OccupancySensor(GridSensor):
    def __init__(
        self,
        limits: np.ndarray = np.array([[-1, -1], [-1, 1], [-1, 1]]),
        resolution: np.ndarray = np.array([10, 10, 10], dtype=int),
        interval: int = -1,
    ):
        super().__init__(limits=limits, resolution=resolution, interval=interval, name="Occupancy")

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
        self._call_counter += 1
        if self._computed and (
            self._interval < 0 or self._call_counter % self._interval != 0
        ):
            return self._grid_values
        start_time = perf_counter()
        distances = self.distances(obstacles)
        self._grid_values = np.array(distances <= 0.0, dtype=int).reshape(self._resolution)
        end_time = perf_counter()

        print(f"Computed Occupancy in {end_time-start_time} s")
        self._computed = True
        return self._grid_values
