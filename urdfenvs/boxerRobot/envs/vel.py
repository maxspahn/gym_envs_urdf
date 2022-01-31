from urdfenvs.boxerRobot.envs.boxerRobotEnv import BoxerRobotEnv


class BoxerRobotVelEnv(BoxerRobotEnv):
    def applyAction(self, action):
        self.robot.apply_base_velocity(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getVelSpaces()
