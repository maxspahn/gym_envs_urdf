import numpy as np
from tiagoReacher.envs.tiagoReacherEnv import TiagoReacherEnv


class TiagoReacherAccEnv(TiagoReacherEnv):

    def applyAction(self, action):
        self.robot.apply_acc_action(action, self.dt())

    def setSpaces(self):
        (self.observation_space, self.action_space) = self.robot.getAccSpaces()
