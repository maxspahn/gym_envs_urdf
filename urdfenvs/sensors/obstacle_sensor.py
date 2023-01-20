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

    Attributes
    ----------

    _observation: dict
        For every object the Pose and Twist are stored in the observation.
        Pose contains position in cartesian format (x, y, z)
        and orientation in quaternion format (x, y, z, w)
        Twist contains velocity in cartesian format (x, y, z)
        and angular velocity in cartesian format (x, y, z)

    """

    def __init__(self):
        super().__init__("ObstacleSensor")
        self._observation = np.zeros(self.get_observation_size())

    def get_obserrvation_size(self):
        """Getter for the dimension of the observation space."""
        size = 0
        for _ in range(2, p.getNumBodies()):
            size += (
                14  # add space for position, velocity,
                    # orientation and angular velocity
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
            spaces_dict[f"obstacle_{obj_id-2}"] = gym.spaces.Dict(
                {
                    "pose": gym.spaces.Dict(
                        {
                            "position": gym.spaces.Box(
                                low=min_os_value,
                                high=max_os_value,
                                shape=(3,),
                                dtype=np.float64,
                            ),
                            "orientation": gym.spaces.Box(
                                low=min_os_value,
                                high=max_os_value,
                                shape=(4,),
                                dtype=np.float64,
                            )
                        }),
                    "twist": gym.spaces.Dict(
                        {
                            "linear": gym.spaces.Box(
                                low=min_os_value,
                                high=max_os_value,
                                shape=(3,),
                                dtype=np.float64,
                            ),
                            "angular": gym.spaces.Box(
                                low=min_os_value,
                                high=max_os_value,
                                shape=(3,),
                                dtype=np.float64,
                            )
                        }
                    )
                }
            )

        return spaces_dict

    def sense(self, robot, *args):
        """
        Sense the exact position of all the objects.


        """
        observation = {}

        # assumption: p.getBodyInfo(0), p.getBodyInfo(1) are the robot and
        # ground plane respectively
        for obj_id in range(2, p.getNumBodies()):
            pos = p.getBasePositionAndOrientation(obj_id)
            vel = p.getBaseVelocity(obj_id)

            observation[f"obstacle_{obj_id-2}"] = {
                "pose": {
                    "position": np.array(pos[0]),
                    "orientation": np.array(pos[1])
                },
                "twist": {
                    "linear": np.array(vel[0]),
                    "angular": np.array(vel[1])
                }
            }

        return observation
