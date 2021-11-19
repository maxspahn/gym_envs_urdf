from tiagoReacher.envs.tiagoReacherEnv import TiagoReacherEnv


class TiagoReacherTorEnv(TiagoReacherEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False, dt=0.01, friction=0.3):
        super().__init__(render=render, dt=dt)
        self._friction = friction

    def reset(self, initialSet=False):
        super().reset(initialSet=initialSet)
        self.robot.disableVelocityControl(self._friction)

    def applyAction(self, action):
        self.robot.apply_base_velocity(action)
        self.robot.apply_torque_action(action)
