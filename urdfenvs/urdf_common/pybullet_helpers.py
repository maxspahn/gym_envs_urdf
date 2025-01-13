import logging
from typing import Tuple, Optional, List

import pybullet


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
        base_orientation = [0.0, 0.0, 0.0, 1.0]
    else:
        base_orientation = orientation[1:] + orientation[:1]
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
        base_position = tuple(base_position[i] for i in range(3))
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
