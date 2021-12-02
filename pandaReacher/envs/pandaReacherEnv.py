from pandaReacher.resources.pandaRobot import PandaRobot
from urdfCommon.urdfEnv import UrdfEnv


class PandaReacherEnv(UrdfEnv):

    def __init__(self, render=False, dt=0.01, friction=0.0, gripper=False, n=7):
        super().__init__(PandaRobot(gripper=gripper, friction=friction), render=render, dt=dt)
        self.setSpaces()
