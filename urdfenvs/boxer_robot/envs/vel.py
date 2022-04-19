from urdfenvs.boxer_robot.envs.boxer_robot_env import BoxerRobotEnv


class BoxerRobotVelEnv(BoxerRobotEnv):
    def apply_action(self, action):
        self._robot.apply_base_velocity(action)

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_velocity_spaces()
