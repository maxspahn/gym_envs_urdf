import numpy as np
from tiagoReacher.envs.tiagoReacherEnv import TiagoReacherEnv


class TiagoReacherAccEnv(TiagoReacherEnv):
    metadata = {"render.modes": ["human"]}

    def applyAction(self, action):
        self.robot.apply_acc_action(action, self.dt())
