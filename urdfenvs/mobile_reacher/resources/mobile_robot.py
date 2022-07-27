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

    def set_joint_names(self):
        if self._gripper:
            mobile_joint_names = ["mobile_joint_x", "mobile_joint_y",
                                  "mobile_joint_theta"]
            panda_joint_names = ["panda_joint"+str(i) for i in range(1,8)]
            self._joint_names = (
                mobile_joint_names+panda_joint_names
            )
        else:
            mobile_joint_names = ["mobile_joint_x", "mobile_joint_y",
                                  "mobile_joint_theta"]
            panda_joint_names = ["panda_joint"+str(i) for i in range(1,8)]
            self._joint_names = (
                mobile_joint_names+panda_joint_names
            )

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
