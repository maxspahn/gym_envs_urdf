from albertReacher.envs.albertReacherEnv import AlbertReacherEnv


class AlbertReacherAccEnv(AlbertReacherEnv):
    def reset(self, initialSet=False, pos=None, vel=None):
        ob = super().reset(initialSet=initialSet, pos=pos, vel=vel)
        self.robot.disableVelocityControl()
        return ob

    def applyAction(self, action):
        self.robot.apply_acc_action(action, self._dt)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getAccSpaces()
