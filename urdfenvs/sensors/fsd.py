"""Module for free space decomposition."""
import numpy as np
from typing import Callable, List


class HalfPlane(object):
    _normal_vector: np.ndarray
    _point: np.ndarray
    _constant: float

    def __init__(self, point: np.ndarray, position: np.ndarray):
        self._normal_vector = position - point
        self._point = point
        self._constant = -np.dot(self.normal(), point)

    def point_behind_plane(self, point: np.ndarray):
        return np.dot(self.normal(), point) + self.constant() <= 0

    def point_infront_plane(self, point: np.ndarray):
        return not self.point_behind_plane(point)

    def normal(self) -> np.ndarray:
        return self._normal_vector

    def point(self) -> np.ndarray:
        return self._point

    def constant(self) -> float:
        return self._constant

    def equation_by_variable(self, variable_name: str) -> Callable:
        if variable_name == "y":
            return (
                lambda x: 1
                / self.normal()[1]
                * (-self.constant() - self.normal()[0] * x)
            )
        return (
            lambda y: 1
            / self.normal()[0]
            * (-self.constant() - self.normal()[1] * y)
        )

    def get_points(self) -> np.ndarray:
        my_fun = self.equation_by_variable("y")
        x = np.arange(0, 2) * 10 - 5
        y_values = my_fun(x)
        if np.any(np.isinf(y_values)):
            x = np.array([self.point()[0], self.point()[0]])
            y = np.array([-5, 5])
            return np.array([x, y])
        return np.array([x, y_values])

    def constraint(self):
        return np.concatenate((self.normal(), np.array([self.constant()])))


def point_to_point_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


class FreeSpaceDecomposition(object):
    _constraints: List[HalfPlane]
    _number_constraints: int
    _max_radius: float
    _position: np.ndarray

    def __init__(
        self,
        position: np.ndarray,
        number_constraints: int = 10,
        max_radius: float = 1.0,
    ):
        self._number_constraints = number_constraints
        self._max_radius = max_radius
        self._constraints = []
        self._position = position

    def set_position(self, position: np.ndarray):
        self._position = position

    def compute_constraints(
        self,
        points: np.ndarray,
    ):
        self._constraints = []
        dists = np.linalg.norm(points - self._position, axis=1)
        idx = np.argsort(dists)
        points = points[idx]
        points = points[dists[idx] < self._max_radius]
        while (
            points.size > 0
            and len(self._constraints) < self._number_constraints
        ):
            point = points[0]
            new_constraint = HalfPlane(point, self._position)
            self._constraints.append(new_constraint)
            mask = np.apply_along_axis(
                new_constraint.point_infront_plane, 1, points
            )
            points = points[mask]

    def constraints(self):
        return self._constraints

    def asdict(self):
        constraint_dict = {}
        for i in range(self._number_constraints):
            if i < len(self._constraints):
                constraint_dict[f"constraint_{i}"] = self._constraints[
                    i
                ].constraint()
            else:
                point = self._position + np.array([20, 20, 0])
                dummy_halfplane = HalfPlane(self._position, point)
                constraint_dict[
                    f"constraint_{i}"
                ] = dummy_halfplane.constraint()
        return constraint_dict

    def get_points(self):
        plot_points = []
        for constraint in self.constraints():
            plot_points.append(constraint.get_points())
        return plot_points
