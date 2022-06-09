import os
import numpy as np
from urdfenvs.urdfCommon.holonomic_robot import HolonomicRobot


class PandaRobot(HolonomicRobot):
    def __init__(self, gripper=False, friction=0.0):
        self._gripper = gripper
        self._friction = friction
        if gripper:
            urdf_file = os.path.join(
                os.path.dirname(__file__), "pandaWithGripper_working.urdf"
            )
            n = 9
        else:
            urdf_file = os.path.join(
                os.path.dirname(__file__), "panda_working.urdf"
            )
            n = 7
        super().__init__(n, urdf_file)

    def set_joint_names(self):
        if self._gripper:
            panda_finger_joint_names = ["panda_finger_joint1",
                                            "panda_finger_joint2"]
            panda_joint_names = ["panda_joint"+str(i)
                                     for i in range(1,8)]
            self._joint_names = (
                panda_joint_names+
                panda_finger_joint_names
            )
        else:
            panda_joint_names = ["panda_joint"+str(i) for i in range(1,8)]
            self._joint_names = (
                panda_joint_names
            )


    def set_acceleration_limits(self):
        acc_limit = np.array(
            [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 1.0, 1.0]
        )
        self._limit_acc_j[0, :] = -acc_limit[0 : self.n()]
        self._limit_acc_j[1, :] = acc_limit[0 : self.n()]
