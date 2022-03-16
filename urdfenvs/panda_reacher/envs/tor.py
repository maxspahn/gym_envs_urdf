from urdfenvs.panda_reacher.envs.panda_reacher_env import PandaReacherEnv


class PandaReacherTorEnv(PandaReacherEnv):
    def reset(self, pos=None, vel=None):
        ob = super().reset(pos=pos, vel=vel)
        self._robot.disable_velocity_control()
        return ob

    def apply_action(self, action):
        self._robot.apply_torque_action(action)

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.getTorqueSpaces()
