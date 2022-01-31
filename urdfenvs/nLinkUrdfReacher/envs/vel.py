from urdfenvs.nLinkUrdfReacher.envs.nLinkUrdfReacherEnv import NLinkUrdfReacherEnv


class NLinkUrdfVelReacherEnv(NLinkUrdfReacherEnv):

    def applyAction(self, action):
        self.robot.apply_vel_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getVelSpaces()
