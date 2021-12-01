from albertReacher.envs.albertReacherEnv import AlbertReacherEnv


class AlbertReacherTorEnv(AlbertReacherEnv):
    metadata = {"render.modes": ["human"]}

    def reset(self, initialSet=False, pos=None, vel=None):
        ob = super().reset(initialSet=initialSet, pos=pos, vel=vel)
        self.robot.disableVelocityControl()
        return ob

    def applyAction(self, action):
        self.robot.apply_base_velocity(action)
        self.robot.apply_torque_action(action)

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getTorqueSpaces()
