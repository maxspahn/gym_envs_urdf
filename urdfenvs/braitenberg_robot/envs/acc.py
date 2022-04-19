from urdfenvs.braitenberg_robot.envs.braitenberg_robot_env import BraitenbergRobotEnv


class BraitenbergRobotAccEnv(BraitenbergRobotEnv):
    def reset(self, pos=None, vel=None):
        ob = super().reset(pos=pos, vel=vel)
        self._robot.disable_velocity_control()
        return ob

    def apply_action(self, action):
        self._robot.apply_acceleration_action(action, self._dt)

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_acceleration_spaces()
