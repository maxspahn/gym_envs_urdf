from urdfenvs.point_robot_urdf.resources.point_robot import PointRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class PointRobotEnv(UrdfEnv):
    def __init__(self, render=False, dt=0.01):
        super().__init__(PointRobot(), render=render, dt=dt)
        self.set_spaces()
        self.reset()
