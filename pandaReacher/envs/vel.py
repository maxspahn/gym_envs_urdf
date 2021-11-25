from pandaReacher.envs.pandaReacherEnv import PandaReacherEnv


class PandaReacherVelEnv(PandaReacherEnv):
    metadata = {"render.modes": ["human"]}

    def applyAction(self, action):
        self.robot.apply_vel_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getVelSpaces()
