from urdfenvs.dual_arm.envs.dual_arm_env import DualArmEnv


class DualArmAccEnv(DualArmEnv):
    def reset(self, pos=None, vel=None):
        ob = super().reset(pos=pos, vel=vel)
        self._robot.disable_velocity_control()
        return ob

    def apply_action(self, action):
        self._robot.apply_acceleration_action(action, self.dt())

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_acceleration_spaces()
