import numpy as np
from urdfenvs.point_robot_urdf.envs.point_robot_env import PointRobotEnv


class PointRobotAccEnv(PointRobotEnv):
    def reset(self, pos=np.zeros(2), vel=np.zeros(2)):
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
