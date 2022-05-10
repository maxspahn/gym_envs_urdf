import pybullet as p
import os
import numpy as np

from urdfenvs.urdfCommon.holonomic_robot import HolonomicRobot


class MobileRobot(HolonomicRobot):
    def __init__(self, gripper=False):
        self._gripper = gripper
        if gripper:
            urdf_file = os.path.join(
                os.path.dirname(__file__), "mobilePandaWithGripper.urdf"
            )
            n = 12
        else:
            urdf_file = os.path.join(
                os.path.dirname(__file__), "mobilePanda.urdf"
            )
            n = 10
        super().__init__(n, urdf_file)

    def set_joint_indices(self):
        if self._gripper:
            self._robot_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
            self._urdf_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
        else:
            self._robot_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
            self._urdf_joints = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]

    def reset(self, pos=None, vel=None):
        self._integrated_velocities = vel
        return super().reset(pos=pos, vel=vel)

    def set_acceleration_limits(self):
        acc_limit = np.array(
            [1.0, 1.0, 1.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 1.0, 1.0]
        )
        self._limit_acc_j[0, :] = -acc_limit[0 : self._n]
        self._limit_acc_j[1, :] = acc_limit[0 : self._n]

    def apply_acceleration_action(self, accs, dt):
        self._integrated_velocities += dt * accs
        self.apply_velocity_action(self._integrated_velocities)

    def move_gripper(self, gripper_vel):
        # TODO Why can't I use velocity control here..
        for i in range(2):
            p.setJointMotorControl2(
                self._robot,
                self._gripper_joints[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=gripper_vel,
            )
