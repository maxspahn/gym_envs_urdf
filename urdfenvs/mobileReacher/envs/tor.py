from urdfenvs.mobileReacher.envs.mobileReacherEnv import MobileReacherEnv


class MobileReacherTorEnv(MobileReacherEnv):

    def reset(self, pos=None, vel=None):
        ob = super().reset(pos=pos, vel=vel)
        self.robot.disableVelocityControl()
        return ob

    def applyAction(self, action):
        self.robot.apply_torque_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getTorqueSpaces()
