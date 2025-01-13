import numpy as np
import pybullet

from urdfenvs.sensors.occupancy_sensor import OccupancySensor


class OccupancySensorPybullet(OccupancySensor):
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
            voxel_positions_chunk = voxel_positions[i : min(i + 16, nb_occupied_cells)]
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
            pybullet.changeVisualShape(bullet_id, -1, rgbaColor=[0.0, 0.0, 0.0, 0.3])
            self._bullet_ids.append(bullet_id)

