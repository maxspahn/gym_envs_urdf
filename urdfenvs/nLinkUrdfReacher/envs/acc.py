from urdfenvs.nLinkUrdfReacher.envs.nLinkUrdfReacherEnv import NLinkUrdfReacherEnv


class NLinkUrdfAccReacherEnv(NLinkUrdfReacherEnv):

    def reset(self, pos=None, vel=None):
        ob = super().reset(pos=pos, vel=vel)
        self.robot.disableVelocityControl()
        return ob

    def applyAction(self, action):
        self.robot.apply_acceleration_action(action, self.dt())

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getAccelerationSpaces()

