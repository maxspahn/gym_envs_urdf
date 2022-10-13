from urdfenvs.sensors.sensor import Sensor
import numpy as np
import pybullet as p
from gym import spaces
from typing import List


class FullSensor(Sensor):
    def __init__(self, goal_mask: list, obstacle_mask: list, variance: int= 0.1):
        self._obstacle_mask = obstacle_mask
        self._goal_mask = goal_mask
        self._name = "FullSensor"
        self._noise_variance = variance

    def _reset(self):
        pass

    def observation_size(self) -> tuple:
        return 1, 2

    def get_observation_space(self):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.observation_size(),
            dtype=np.float64,
        )

    def sense(self, robot, obst_ids: List[int], goal_ids: List[int]):
        sensor_observation = {"goals": [], "obstacles": []}

        exact_observations = []
        for obst_id in obst_ids:
            exact_observation = []

            if 'position' in self._obstacle_mask:
                position, orientation = p.getBasePositionAndOrientation(obst_id)
                exact_observation.append(np.array(position))

            if 'velocity' in self._obstacle_mask:
                linear, angular = p.getBaseVelocity(obst_id)
                exact_observation.append(np.array(linear))

            if 'radius' in self._obstacle_mask:
                shape_data = p.getCollisionShapeData(obst_id, -1)[0]
                assert shape_data[2] == p.GEOM_SPHERE

                exact_observation.append(float(shape_data[3][0]))

            exact_observations.append(exact_observation)

        noisy_observations = []
        for exact_observation in exact_observations:
            if isinstance(exact_observation, np.ndarray):
                noisy_observation = np.random.normal(exact_observation, self._noise_variance)
            else:
                noisy_observation = exact_observation
            noisy_observations.append(noisy_observation)
        sensor_observation["obstacles"] = noisy_observations


        exact_observations = []
        for goal_id in goal_ids:
            exact_observation = []

            if 'position' in self._goal_mask:
                position, orientation = p.getBasePositionAndOrientation(goal_id)
                exact_observation.append(np.array(position))

            if 'velocity' in self._goal_mask:
                linear, angular = p.getBaseVelocity(goal_id)
                exact_observation.append(np.array(linear))

            exact_observations.append(exact_observation)

        noisy_observations = []
        for exact_observation in exact_observations:
            if isinstance(exact_observation, np.ndarray):
                noisy_observation = np.random.normal(exact_observation, self._noise_variance)
            else:
                noisy_observation = exact_observation
            noisy_observations.append(noisy_observation)

        sensor_observation["goals"] = noisy_observations

        return sensor_observation
