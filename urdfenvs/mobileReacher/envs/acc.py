from urdfenvs.mobileReacher.envs.mobileReacherEnv import MobileReacherEnv


class MobileReacherAccEnv(MobileReacherEnv):

    def reset(self, initialSet=False, pos=None, vel=None):
        ob = super().reset(initialSet=initialSet, pos=pos, vel=vel)
        self.robot.disableVelocityControl()
        return ob

    def applyAction(self, action):
        self.robot.apply_acc_action(action, self.dt())

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getAccSpaces()
