from urdfenvs.panda_reacher.envs.panda_reacher_env import PandaReacherEnv


class PandaReacherVelEnv(PandaReacherEnv):
    metadata = {"render.modes": ["human"]}

    def apply_action(self, action):
        self._robot.apply_velocity_action(action)

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_velocity_spaces()
