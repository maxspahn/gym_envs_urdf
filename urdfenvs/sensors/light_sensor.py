"""Module for light sensor simulation."""
import numpy as np
import pybullet as p
import gym

from urdfenvs.sensors.sensor import Sensor

from urdfenvs.helpers.functions import link_name_to_link_id


class LightSensor(Sensor):
    """
    The light sensor senses the intensity of a light source as a scalar value.

    Attributes
    ----------

    _link_name: str
        Link of robot where the sensor should be connected to.
    """

    def __init__(self, link_name: str):
        super().__init__("lightSensor")
        self._link_name = link_name

    def get_observation_size(self):
        """Getter for the dimension of the observation space."""
        return 1

    def get_observation_space(self):
        """Create observation space, all observations should be inside the
        observation space."""
        return gym.spaces.Box(
            0,
            1,
            shape=(self.get_observation_size(),),
            dtype=np.float64,
        )

    def get_position_of_light_source(self):
        """Senses the exact position of the light source.

        Assumes that there is no other obstacle in the environment.
        Link 0 and 1 are reserved to robot and ground plane.

        Returns
        ---------
        np.array
            Position of the light source as (x, y)
        """
        for obj_id in range(2, p.getNumBodies()):
            pos = p.getBasePositionAndOrientation(obj_id)[0][0:2]
            vel = p.getBaseVelocity(obj_id)
        return np.array(pos)

    def get_pose_of_sensor(self):
        """Senses the exact position of the light source.

        Assumes that there is no other obstacle in the environment.
        Link 0 and 1 are reserved to robot and ground plane.

        Returns
        -----------
        np.array
            Pose of the sensor as (x, y, theta)
        """
        link_id = link_name_to_link_id(0, self._link_name)
        link_state = p.getLinkState(0, link_id)
        position_sensor = link_state[0][0:2]
        orientation_sensor = p.getEulerFromQuaternion(link_state[1])[2]
        # make sure that the rotation is within -pi and pi
        orientation_sensor -= np.pi / 2.0
        if orientation_sensor < -np.pi:
            orientation_sensor += 2 * np.pi
        pose_sensor = np.array([position_sensor[0], position_sensor[1], orientation_sensor])
        return pose_sensor

    def sense(self, robot):
        pos_light_source = self.get_position_of_light_source()
        pos_sensor = self.get_position_of_sensor()
        print(f"pos_light_source: {pos_light_source}")
        print(f"pos_sensor: {pos_sensor}")
        # TODO: Insert function to compute the intensity of the light source <19-04-22, maxspahn> #
        return 0.5
