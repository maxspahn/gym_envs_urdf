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

    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        sensor_observation = {"goals": [], "obstacles": []}

        exact_observations = []
        for obst_id, obstacle in obstacles.items():
            exact_observation = []
            for mask_item in self._obstacle_mask:
                if mask_item == 'position':
                    value, _ = p.getBasePositionAndOrientation(obst_id)

                elif mask_item == 'velocity':
                    value, _ = p.getBaseVelocity(obst_id)

                else:
                    try:
                        value = getattr(obstacle, mask_item)(t=t)
                    except TypeError:
                        value = getattr(obstacle, mask_item)()
                exact_observation.append(np.array(value))
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
        for goal_id, goal in goals.items():
            exact_observation = []
            for mask_item in self._goal_mask:
                if mask_item == 'position':
                    value, _ = p.getBasePositionAndOrientation(goal_id)

                elif mask_item == 'velocity':
                    value, _ = p.getBaseVelocity(goal_id)
                else:
                    try:
                        value = getattr(goal, mask_item)(t=t)
                    except TypeError:
                        value = getattr(goal, mask_item)()
                exact_observation.append(np.array(value))
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
