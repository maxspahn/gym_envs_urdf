import logging
from typing import Tuple, Optional, List

import numpy as np
import pybullet


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


class LinkIdNotFoundError(Exception):
    pass


def extract_link_id(robot, link_name: str):
    number_links = pybullet.getNumJoints(robot)
    joint_names = []
    for i in range(number_links):
        joint_name = pybullet.getJointInfo(robot, i)[1].decode("UTF-8")
        joint_names.append(joint_name)
        if joint_name == link_name:
            return i
    raise LinkIdNotFoundError(
        f"Link with name {link_name} not found. "
        f"Possible links are {joint_names}"
    )


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


def add_shape(
    shape_type: str,
    size: list,
    color: Optional[List[float]] = None,
    movable: bool = False,
    orientation: Optional[Tuple[float]] = None,
    position: Optional[Tuple[float]] = None,
    scaling: float = 1.0,
    urdf: Optional[str] = None,
    with_collision_shape: bool = True,
) -> int:

    mass = float(movable)
    if color is None:
        color = [1.0, 1.0, 1.0, 1.0]
    if orientation is None:
        base_orientation = (0.0, 0.0, 0.0, 1.0)
    else:
        base_orientation = orientation
    if position is None:
        base_position = (0.0, 0.0, 0.0)
    else:
        base_position = position
    if shape_type in ["sphere", "splineSphere", "analyticSphere"]:
        shape_id = pybullet.createCollisionShape(
            pybullet.GEOM_SPHERE, radius=size[0]
        )
        visual_shape_id = pybullet.createVisualShape(
            pybullet.GEOM_SPHERE,
            rgbaColor=color,
            specularColor=[1.0, 0.5, 0.5],
            radius=size[0],
        )

    elif shape_type == "box":
        half_extens = [s / 2 for s in size]
        base_position = tuple(base_position[i] - size[i] for i in range(3))
        shape_id = pybullet.createCollisionShape(
            pybullet.GEOM_BOX, halfExtents=half_extens
        )
        visual_shape_id = pybullet.createVisualShape(
            pybullet.GEOM_BOX,
            rgbaColor=color,
            specularColor=[1.0, 0.5, 0.5],
            halfExtents=half_extens,
        )

    elif shape_type == "cylinder":
        shape_id = pybullet.createCollisionShape(
            pybullet.GEOM_CYLINDER, radius=size[0], height=size[1]
        )
        visual_shape_id = pybullet.createVisualShape(
            pybullet.GEOM_CYLINDER,
            rgbaColor=color,
            specularColor=[1.0, 0.5, 0.5],
            radius=size[0],
            length=size[1],
        )

    elif shape_type == "capsule":
        shape_id = pybullet.createCollisionShape(
            pybullet.GEOM_CAPSULE, radius=size[0], height=size[1]
        )
        visual_shape_id = pybullet.createVisualShape(
            pybullet.GEOM_CAPSULE,
            rgbaColor=color,
            specularColor=[1.0, 0.5, 0.5],
            radius=size[0],
            length=size[1],
        )
    elif shape_type == "urdf":
        shape_id = pybullet.loadURDF(
            fileName=urdf, basePosition=base_position, globalScaling=scaling
        )
        return shape_id
    else:
        logging.warning("Unknown shape type: {shape_type}, aborting...")
        return -1
    if not with_collision_shape:
        shape_id = -1
    bullet_id = pybullet.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=base_position,
        baseOrientation=base_orientation,
    )
    return bullet_id
