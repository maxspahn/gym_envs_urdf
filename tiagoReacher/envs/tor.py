import numpy as np
from tiagoReacher.envs.tiagoReacherEnv import TiagoReacherEnv


class TiagoReacherTorEnv(TiagoReacherEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False, dt=0.01, n=19, friction=0.3):
        super().__init__(render=render, n=n, dt=dt)
        self._friction = friction

    def reset(self, pos=np.zeros(20), vel=np.zeros(19)):
        ob = super().reset(pos=pos, vel=vel)
        self.robot.disableVelocityControl(self._friction)
        return ob

    def applyAction(self, action):
        self.robot.apply_base_velocity(action)
        self.robot.apply_torque_action(action)
