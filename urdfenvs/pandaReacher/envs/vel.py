from urdfenvs.pandaReacher.envs.pandaReacherEnv import PandaReacherEnv


class PandaReacherVelEnv(PandaReacherEnv):
    metadata = {"render.modes": ["human"]}

    def applyAction(self, action):
        self.robot.apply_velocity_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getVelocitySpaces()
