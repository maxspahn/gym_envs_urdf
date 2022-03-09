from urdfenvs.mobileReacher.envs.mobileReacherEnv import MobileReacherEnv


class MobileReacherVelEnv(MobileReacherEnv):
    def applyAction(self, action):
        self.robot.apply_velocity_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getVelocitySpaces()

