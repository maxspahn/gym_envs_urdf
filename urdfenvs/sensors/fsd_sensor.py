"""Module for fsd sensor based on lidar."""
import numpy as np
#import pybullet
import gymnasium as gym

from urdfenvs.sensors.sensor import Sensor
from urdfenvs.sensors.fsd import FreeSpaceDecomposition
from urdfenvs.urdf_common.helpers import quaternion_between_vectors


class FSDSensor(Sensor):
    def __init__(
        self,
        max_radius: float = 1.0,
        number_constraints: int = 10,
        plotting_interval_fsd: int = -1,
        variance: float = 0.0,
        planar_visualization: bool = True,
        physics_engine_name: str = 'pybullet'
    ):
        Sensor.__init__(
            self,
            "AbstractFSDSensor",
            variance=variance,
            physics_engine_name=physics_engine_name,
        )
        self._fsd = FreeSpaceDecomposition(
            np.array([0.0, 0.0, 0.0]),
            max_radius=max_radius,
            number_constraints=number_constraints,
        )
        self._plotting_interval_fsd = plotting_interval_fsd
        self._plane_ids = []
        self._planar_visualization = planar_visualization

    def get_observation_space(self, obstacles: dict, goals: dict):
        """Create observation space, all observations should be inside the
        observation space."""
        observation_space = {}
        for i in range(self._fsd._number_constraints):
            observation_space[f"constraint_{i}"] = gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(4,),
                dtype=float,
            )
        return gym.spaces.Dict({self._name: gym.spaces.Dict(observation_space)})

    def compute_fsd(
        self, point_positions: np.ndarray, center_position: np.ndarray
    ):
        self._fsd.set_position(center_position)
        self._fsd.compute_constraints(point_positions)
        if (
            self._plotting_interval_fsd > 0
            and self._call_counter % self._plotting_interval_fsd == 0
        ):
            if self._planar_visualization:
                self.visualize_constraints()
            else:
                self.visualize_constraints_with_boxes(center_position)
        return self._fsd.asdict()

    def visualize_constraints(self):
        plot_points = self._fsd.get_points()
        self._physics_engine.clear_visualizations()
        for plot_point in plot_points:
            start_point = (plot_point[0, 0], plot_point[-1, 0], self._height)
            end_point = (plot_point[0, 1], plot_point[-1, 1], self._height)
            self._physics_engine.add_visualization_line(start_point, end_point)
