from urdfenvs.generic_reacher.envs.generic_reacher_env import GenericReacherEnv


class GenericReacherVelEnv(GenericReacherEnv):
    metadata = {"render.modes": ["human"]}

    def apply_action(self, action):
        self._robot.apply_velocity_action(action)

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_velocity_spaces()
