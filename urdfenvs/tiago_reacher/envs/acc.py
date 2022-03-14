from urdfenvs.tiago_reacher.envs.tiago_reacher_env import TiagoReacherEnv


class TiagoReacherAccEnv(TiagoReacherEnv):
    def apply_action(self, action):
        self._robot.apply_acceleration_action(action, self.dt())

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_acceleration_spaces()
