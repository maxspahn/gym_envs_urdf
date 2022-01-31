from pointRobotUrdf.envs.pointRobotEnv import PointRobotEnv


class PointRobotVelEnv(PointRobotEnv):

    def applyAction(self, action):
        self.robot.apply_vel_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getVelSpaces()
