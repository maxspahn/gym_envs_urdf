from typing import Tuple

import numpy as np
import gymnasium as gym

class WrongObservationError(Exception):
    pass


class WrongActionError(Exception):
    pass


def check_observation(obs, ob):
    for key, value in ob.items():
        if isinstance(value, dict):
            check_observation(obs[key], value)
        elif isinstance(value, np.ndarray):
            if isinstance(obs[key], gym.spaces.Discrete):
                continue
            if not obs[key].contains(value):
                s = f"key: {key}: {value} not in {obs[key]}"
                if np.any(value < obs[key].low):
                    index = np.where(value < obs[key].low)[0]
                    value_at_index = value[index]
                    s += f"\nAt index {index.tolist()}: {value_at_index} < {obs[key].low[index]}"
                if np.any(value > obs[key].high):
                    index = np.where(value > obs[key].high)[0]
                    value_at_index = value[index]
                    s += f"\nAt index {index.tolist()}: {value_at_index} > {obs[key].high[index]}"

                raise WrongObservationError(s)
        else:
            raise Exception(f"Observation checking failed for key:{key} value:{value}.")



def quaternion_between_vectors(v1, v2, ordering="wxyz"):
    quaternion = np.zeros(4)
    a_part = np.cross(v1, v2)
    if ordering == "wxyz":
        quaternion[0] = np.sqrt(
            (np.linalg.norm(v1) ** 2) * (np.linalg.norm(v2) ** 2)
        ) + np.dot(v1, v2)
        quaternion[1:4] = a_part
    elif ordering == "xyzw":
        quaternion[0:3] = a_part
        quaternion[3] = np.sqrt(
            (np.linalg.norm(v1) ** 2) * (np.linalg.norm(v2) ** 2)
        ) + np.dot(v1, v2)
    else:
        raise InvalidQuaternionOrderError(
            f"Order {ordering} is not permitted, options are 'xyzw', and 'wxyz'"
        )
    normed_quat = quaternion / np.linalg.norm(quaternion)
    return normed_quat



class InvalidQuaternionOrderError(Exception):
    pass


def quaternion_to_rotation_matrix(
    quaternion: np.ndarray, ordering: str = "wxyz"
) -> np.ndarray:
    # Normalize the quaternion if needed
    quaternion /= np.linalg.norm(quaternion)

    if ordering == "wxyz":
        w, x, y, z = quaternion
    elif ordering == "xyzw":
        x, y, z, w = quaternion
    else:
        raise InvalidQuaternionOrderError(
            f"Order {ordering} is not permitted, options are 'xyzw', and 'wxyz'"
        )
    rotation_matrix = np.array(
        [
            [
                1 - 2 * y**2 - 2 * z**2,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
            ],
            [
                2 * x * y + 2 * w * z,
                1 - 2 * x**2 - 2 * z**2,
                2 * y * z - 2 * w * x,
            ],
            [
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x**2 - 2 * y**2,
            ],
        ]
    )

    return rotation_matrix


def get_transformation_matrix(
    quaternion: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(quaternion, ordering="xyzw")

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation

    return transformation_matrix


def matrix_to_quaternion(matrix, ordering="wxyz") -> Tuple[float]:
    """
    Convert a 4x4 transformation matrix to a quaternion.

    Parameters:
        matrix (numpy.ndarray): The 4x4 transformation matrix.

    Returns:
        numpy.ndarray: The quaternion representation (w, x, y, z).
    """

    # Extract the rotation matrix from the transformation matrix
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]

    # Calculate the trace of the rotation matrix
    trace = np.trace(rotation_matrix)

    if trace > 0:
        # The quaternion calculation when the trace is positive
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    elif (rotation_matrix[0, 0] > rotation_matrix[1, 1]) and (
        rotation_matrix[0, 0] > rotation_matrix[2, 2]
    ):
        # The quaternion calculation when the trace is largest along x-axis
        s = (
            np.sqrt(
                1.0
                + rotation_matrix[0, 0]
                - rotation_matrix[1, 1]
                - rotation_matrix[2, 2]
            )
            * 2
        )
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        x = 0.25 * s
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        # The quaternion calculation when the trace is largest along y-axis
        s = (
            np.sqrt(
                1.0
                + rotation_matrix[1, 1]
                - rotation_matrix[0, 0]
                - rotation_matrix[2, 2]
            )
            * 2
        )
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        y = 0.25 * s
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        # The quaternion calculation when the trace is largest along z-axis
        s = (
            np.sqrt(
                1.0
                + rotation_matrix[2, 2]
                - rotation_matrix[0, 0]
                - rotation_matrix[1, 1]
            )
            * 2
        )
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        z = 0.25 * s

    quaternion = np.array([1, 0, 0, 0])
    if ordering == "wxyz":
        quaternion = np.array([w, x, y, z])
    elif ordering == "xyzw":
        quaternion = np.array([x, y, z, w])
    else:
        raise InvalidQuaternionOrderError(
            f"Order {ordering} is not permitted, options are 'xyzw', and 'wxyz'"
        )

    return translation, quaternion


