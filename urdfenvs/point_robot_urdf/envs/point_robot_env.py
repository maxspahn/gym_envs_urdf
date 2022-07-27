from urdfenvs.point_robot_urdf.resources.point_robot import PointRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class PointRobotEnv(UrdfEnv):
    def __init__(self, **kwargs):
        super().__init__(PointRobot(), **kwargs)
        self.reset()
