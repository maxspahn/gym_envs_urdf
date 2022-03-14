import numpy as np
import pybullet as p
import gym
from urdfenvs.sensors.sensor import Sensor


class ObstacleSensor(Sensor):
    """
    the ObstacleSensor class is a sensor sensing the exact position of every
    object. The ObstacleSensor is thus a full information sensor which in the
    real world can never exist. The ObstacleSensor returns a dictionary with
    the position of every object when the sense function is called.
    """

    def __init__(self):
        super().__init__("obstacleSensor")
        self._observation = np.zeros(self.get_observation_size())

    def get_obserrvation_size(self):
        """Getter for the dimension of the observation space."""
        size = 0
        for _ in range(2, p.getNumBodies()):
            size += (
                12  # add space for x, xdot, theta and thetadot for every object
            )
        return size

    def get_observation_space(self):
        """
        Create observation space, all observed objects should be inside the
        observation space.
        """
        spaces_dict = gym.spaces.Dict()

        min_os_value = -1000
        max_os_value = 1000

        for obj_id in range(2, p.getNumBodies()):
            spaces_dict[str(obj_id)] = gym.spaces.Dict(
                {
                    "x": gym.spaces.Box(
                        low=min_os_value,
                        high=max_os_value,
                        shape=(3,),
                        dtype=np.float64,
                    ),
                    "xdot": gym.spaces.Box(
                        low=min_os_value,
                        high=max_os_value,
                        shape=(3,),
                        dtype=np.float64,
                    ),
                    "theta": gym.spaces.Box(
                        low=-2 * np.pi,
                        high=2 * np.pi,
                        shape=(3,),
                        dtype=np.float64,
                    ),
                    "thetadot": gym.spaces.Box(
                        low=min_os_value,
                        high=max_os_value,
                        shape=(3,),
                        dtype=np.float64,
                    ),
                }
            )

        return spaces_dict

    def sense(self, robot):
        """Sense the exact position of all the objects."""
        observation = {}

        # assumption: p.getBodyInfo(0), p.getBodyInfo(1) are the robot and
        # ground plane respectively
        for obj_id in range(2, p.getNumBodies()):

            pos = p.getBasePositionAndOrientation(obj_id)
            vel = p.getBaseVelocity(obj_id)

            observation[str(obj_id)] = {
                "x": np.array(pos[0]),
                "xdot": np.array(vel[0]),
                "theta": np.array(p.getEulerFromQuaternion(pos[1])),
                "thetadot": np.array(vel[1]),
            }

        return observation
