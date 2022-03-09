from urdfenvs.tiagoReacher.envs.tiagoReacherEnv import TiagoReacherEnv


class TiagoReacherVelEnv(TiagoReacherEnv):

    def applyAction(self, action):
        self.robot.apply_base_velocity(action)
        self.robot.apply_velocity_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getVelocitySpaces()
