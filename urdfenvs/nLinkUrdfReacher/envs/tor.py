from urdfenvs.nLinkUrdfReacher.envs.nLinkUrdfReacherEnv import NLinkUrdfReacherEnv


class NLinkUrdfTorReacherEnv(NLinkUrdfReacherEnv):

    def reset(self, initialSet=False, pos=None, vel=None):
        ob = super().reset(initialSet=initialSet, pos=pos, vel=vel)
        self.robot.disableVelocityControl()
        return ob

    def applyAction(self, action):
        self.robot.apply_torque_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getTorqueSpaces()
