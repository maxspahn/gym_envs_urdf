import pybullet as p
import os
import numpy as np

from urdfCommon.abstractRobot import AbstractRobot


class MobileRobot(AbstractRobot):
    def __init__(self, gripper=False):
        self._gripper = gripper
        if gripper:
            fileName = os.path.join(os.path.dirname(__file__), 'mobilePandaWithGripper.urdf')
            n = 12
        else:
            fileName = os.path.join(os.path.dirname(__file__), 'mobilePanda.urdf')
            n = 10
        super().__init__(n, fileName)

    def setJointIndices(self):
        if self._gripper:
            self.robot_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
            self.urdf_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
        else:
            self.robot_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
            self.urdf_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]

    def reset(self, pos=None, vel=None):
        self._vels_int = vel
        return super().reset(pos=pos, vel=vel)

    def setAccLimits(self):
        accLimit = np.array([1.0, 1.0, 1.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 1.0, 1.0])
        self._limitAcc_j[0, :] = -accLimit[0:self._n]
        self._limitAcc_j[1, :] = accLimit[0:self._n]

    def apply_acc_action(self, accs, dt):
        self._vels_int += dt * accs
        self.apply_vel_action(self._vels_int)

    def moveGripper(self, gripperVel):
        # TODO Why can't I use velocity control here..
        for i in range(2):
            p.setJointMotorControl2(self.robot, self.gripper_joints[i],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=gripperVel)
