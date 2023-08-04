"""Module for occupancy sensor simulation."""
from time import perf_counter
import logging

import numpy as np
import pybullet
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
    ):
        super().__init__(
            limits=limits,
            resolution=resolution,
            interval=interval,
            name="Occupancy",
            variance=variance,
            plotting_interval=plotting_interval,
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

    def update_occupancy_visualization(self):
        """
        Updates the position of the boxes visualizing the occupancy.
        If the boxes have not been initialized, the function
        init_occupancy_visualization(self) is called.

        Parameters
        ------------
        """
        for bullet_id in self._bullet_ids:
            pybullet.removeBody(bullet_id)
        self._bullet_ids = []
        grid_values_flat = self._grid_values.reshape((-1, 1))
        voxel_positions = []
        for voxel_id in range(self.number_of_voxels()):
            if grid_values_flat[voxel_id] == 1:
                voxel_positions.append(self._mesh_flat[voxel_id].tolist())

        nb_occupied_cells = len(voxel_positions)
        for i in range(0, nb_occupied_cells, 16):
            voxel_positions_chunk = voxel_positions[
                i : min(i + 16, nb_occupied_cells)
            ]
            half_extens = np.tile(
                self._voxel_size * 0.5, (nb_occupied_cells, 1)
            ).tolist()
            shape_types = [pybullet.GEOM_BOX] * len(voxel_positions_chunk)
            half_extens = np.tile(
                self._voxel_size * 0.5, (len(voxel_positions_chunk), 1)
            ).tolist()
            visual_shape_id = pybullet.createVisualShapeArray(
                shape_types,
                halfExtents=half_extens,
                visualFramePositions=voxel_positions_chunk,
            )
            bullet_id = pybullet.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                useMaximalCoordinates=False,
            )
            pybullet.changeVisualShape(
                bullet_id, -1, rgbaColor=[0.0, 0.0, 0.0, 0.3]
            )
            self._bullet_ids.append(bullet_id)
