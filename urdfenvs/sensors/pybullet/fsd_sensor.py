import numpy as np
import pybullet

from urdfenvs.urdf_common.pybullet_helpers import add_shape
from urdfenvs.sensors.fsd_sensor import FSDSensor


class FSDSensorPybullet(FSDSensor):
    def visualize_constraints_with_boxes(self, center_position: np.ndarray):
        constraints = self._fsd.constraints()
        positions = []
        orientations = []
        half_extens = []
        shape_types = []
        for constraint in constraints:
            normal = constraint.normal()
            point = constraint.point()
            vector = np.array([1.0, 0.0, 0.0])
            orientation = quaternion_between_vectors(
                vector, normal, ordering="xyzw",
            )
            orientations.append(orientation)
            shape_types.append(pybullet.GEOM_BOX)
            half_extens.append([0.02, 10.0, 10.0])
            positions.append(point)

        for plane_id in self._plane_ids:
            pybullet.removeBody(plane_id)
        self._plane_ids = []
        visual_shape_id = pybullet.createVisualShapeArray(
            shape_types,
            halfExtents=half_extens,
            visualFramePositions=positions,
            visualFrameOrientations=orientations,
        )
        bullet_id = pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape_id,
            useMaximalCoordinates=False,
        )
        pybullet.changeVisualShape(
            bullet_id, -1, rgbaColor=[0.0, 0.0, 0.0, 0.1]
        )
        self._plane_ids.append(bullet_id)

        body_id_sphere = add_shape(
            "sphere",
            size=[0.15],
            color=[1.0, 0.0, 0.0, 0.3],
            position=center_position,
            with_collision_shape=False,
        )
        self._plane_ids.append(body_id_sphere)


    