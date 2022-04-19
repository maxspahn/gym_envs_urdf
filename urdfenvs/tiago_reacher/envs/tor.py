import numpy as np
from urdfenvs.tiago_reacher.envs.tiago_reacher_env import TiagoReacherEnv


class TiagoReacherTorEnv(TiagoReacherEnv):
    def __init__(self, render=False, dt=0.01, n=19, friction=0.3):
        super().__init__(render=render, dt=dt)
        self._friction = friction

    def reset(self, pos=np.zeros(20), vel=np.zeros(19)):
        ob = super().reset(pos=pos, vel=vel)
        self._robot.disable_velocity_control(self._friction)
        return ob

    def apply_action(self, action):
        self._robot.apply_base_velocity(action)
        self._robot.apply_torque_action(action)

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_torque_spaces()
