from urdfenvs.sensors.sensor import Sensor
import numpy as np
import pybullet as p
from gymnasium import spaces


class FullSensor(Sensor):
    def __init__(
        self, goal_mask: list, obstacle_mask: list, variance: float = 0.0
    ):
        super().__init__("FullSensor", variance=variance)
        self._obstacle_mask = obstacle_mask
        self._goal_mask = goal_mask

    def _reset(self):
        pass

    def observation_size(self) -> tuple:
        return 1, 2

    def get_observation_space(self, obstacles: dict, goals: dict):
        observation_space = {}
        observation_space_obstacles = {}
        for obst_id, obstacle in obstacles.items():
            observation_space_obstacle = {}
            if "position" in self._obstacle_mask:
                observation_space_obstacle["position"] = spaces.Box(
                    low=np.array([-50, -50, -50]),
                    high=np.array([50, 50, 50]),
                    dtype=float,
                )
            if "velocity" in self._obstacle_mask:
                observation_space_obstacle["velocity"] = spaces.Box(
                    low=np.array([-50, -50, -50]),
                    high=np.array([50, 50, 50]),
                    dtype=float,
                )
            if "acceleration" in self._obstacle_mask:
                observation_space_obstacle["acceleration"] = spaces.Box(
                    low=np.array([-5, -5, -5]),
                    high=np.array([5, 5, 5]),
                    dtype=float,
                )
            if "type" in self._obstacle_mask:
                observation_space_obstacle["type"] = spaces.Discrete(128)
            if "size" in self._obstacle_mask:
                low_limit_size = [
                    0,
                ] * len(obstacle.size())
                high_limit_size = [
                    50,
                ] * len(obstacle.size())
                observation_space_obstacle["size"] = spaces.Box(
                    low=np.array(low_limit_size),
                    high=np.array(high_limit_size),
                    dtype=float,
                )
            observation_space_obstacles[obst_id] = spaces.Dict(
                observation_space_obstacle
            )
        if observation_space_obstacles:
            observation_space["obstacles"] = spaces.Dict(
                observation_space_obstacles
            )
        observation_space_goals = {}
        for goal_id, goal in goals.items():
            observation_space_goal = {}
            if "position" in self._goal_mask:
                observation_space_goal["position"] = spaces.Box(
                    low=np.array([-50, -50, -50]),
                    high=np.array([50, 50, 50]),
                    dtype=float,
                )
            if "velocity" in self._goal_mask:
                observation_space_goal["velocity"] = spaces.Box(
                    low=np.array([-50, -50, -50]),
                    high=np.array([50, 50, 50]),
                    dtype=float,
                )
            if "acceleration" in self._goal_mask:
                observation_space_goal["acceleration"] = spaces.Box(
                    low=np.array([-5, -5, -5]),
                    high=np.array([5, 5, 5]),
                    dtype=float,
                )
            if "weight" in self._goal_mask:
                observation_space_goal["weight"] = spaces.Box(
                    low=np.array([0]),
                    high=np.array([10]),
                    dtype=float,
                )
            if "is_primary_goal" in self._goal_mask:
                observation_space_goal["is_primary_goal"] = spaces.Box(
                    low=np.array([False], dtype=bool),
                    high=np.array([True], dtype=bool),
                    dtype=bool,
                )
            observation_space_goals[goal_id] = spaces.Dict(
                observation_space_goal
            )
        if observation_space_goals:
            observation_space["goals"] = spaces.Dict(observation_space_goals)
        return spaces.Dict({self._name: spaces.Dict(observation_space)})

    def sense(self, robot, obstacles: dict, goals: dict, t: float):
        sensor_observation = {}

        observations = {}
        for obst_id, obstacle in obstacles.items():
            observation = {}
            for mask_item in self._obstacle_mask:
                if mask_item == "position":
                    value, _ = p.getBasePositionAndOrientation(obst_id)

                elif mask_item == "velocity":
                    value, _ = p.getBaseVelocity(obst_id)

                else:
                    try:
                        value = getattr(obstacle, mask_item)(t=t)
                    except TypeError:
                        value = getattr(obstacle, mask_item)()
                if isinstance(value, float):
                    value = [value]
                if isinstance(value, str):
                    observation[mask_item] = np.array([ord(c) for c in value])
                else:
                    observation[mask_item] = np.random.normal(
                        np.array(value), self._variance
                    ).astype("float32")
            observations[obst_id] = observation

        if observations:
            sensor_observation["obstacles"] = observations

        observations = {}
        for obst_id, goal in goals.items():
            observation = {}
            for mask_item in self._goal_mask:
                if mask_item == "position":
                    value, _ = p.getBasePositionAndOrientation(obst_id)

                elif mask_item == "velocity":
                    value, _ = p.getBaseVelocity(obst_id)

                else:
                    try:
                        value = getattr(goal, mask_item)(t=t)
                    except TypeError:
                        value = getattr(goal, mask_item)()
                if isinstance(value, float):
                    value = [value]
                if isinstance(value, bool):
                    observation[mask_item] = np.array([value])
                else:
                    observation[mask_item] = np.random.normal(
                        np.array(value), self._variance
                    ).astype("float32")
            observations[obst_id] = observation

        if observations:
            sensor_observation["goals"] = observations

        return sensor_observation
