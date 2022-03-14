from urdfenvs.tiago_reacher.envs.tiago_reacher_env import TiagoReacherEnv


class TiagoReacherVelEnv(TiagoReacherEnv):
    def apply_action(self, action):
        self._robot.apply_base_velocity(action)
        self._robot.apply_velocity_action(action)

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_velocity_spaces()
