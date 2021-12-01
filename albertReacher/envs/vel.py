from albertReacher.envs.albertReacherEnv import AlbertReacherEnv


class AlbertReacherVelEnv(AlbertReacherEnv):
    def applyAction(self, action):
        self.robot.apply_base_velocity(action)
        self.robot.apply_vel_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getVelSpaces()
