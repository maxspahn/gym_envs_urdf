import numpy as np
import pybullet

from urdfenvs.sensors.lidar import Lidar
from urdfenvs.urdf_common.pybullet_helpers import add_shape


class LidarPybullet(Lidar):
    def init_lidar_spheres(self, lidar_position):
        """
        Initialize the Lidar spheres to visualize the sensing in bullet.
        The visual spheres are initialized once and then update in the
        update function. The relative positions are augmented by a third
        column to have a full position, including the z-coordinate.

        Parameters
        ------------
        lidar_position : np.ndarray
            The position of the lidar sensor link.
        """
        q = lidar_position
        q_obs = self._rel_positions.reshape(self._nb_rays, 2)
        q_obs = np.append(q_obs, np.zeros((self._nb_rays, 1)), axis=1)
        for ray_id in range(self._nb_rays):
            body_id_sphere = add_shape(
                "sphere",
                size=[0.05],
                color=[0.0, 0.0, 0.0, 0.8],
                position=q + q_obs[ray_id],
                with_collision_shape=False,
            )
            self._sphere_ids[ray_id] = body_id_sphere

    def update_lidar_spheres(self, lidar_position):
        """
        Updates the position of the spheres visualizing the sensing with lidar.
        If the spheres have not been initialized, the init_lidar_spheres
        function is called.

        Parameters
        ------------
        lidar_position : np.ndarray
            The position of the lidar sensor link.
        """
        if self._sphere_ids[0] == -1:
            self.init_lidar_spheres(lidar_position)
        q = lidar_position
        # Reshape and add z-values to the sensor data.
        q_obs = self._rel_positions.reshape(self._nb_rays, 2)
        q_obs = np.append(q_obs, np.zeros((self._nb_rays, 1)), axis=1)
        for ray_id in range(self._nb_rays):
            pybullet.resetBasePositionAndOrientation(
                int(self._sphere_ids[ray_id]), q + q_obs[ray_id], [0, 0, 0, 1]
            )
