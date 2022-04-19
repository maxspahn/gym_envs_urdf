from urdfenvs.prius.envs.prius_env import PriusEnv


class PriusVelEnv(PriusEnv):
    def apply_action(self, action):
        self._robot.apply_velocity_action(action)

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_velocity_spaces()
