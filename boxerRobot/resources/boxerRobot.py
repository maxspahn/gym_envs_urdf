import pybullet as p
import os
import numpy as np

from urdfCommon.abstractRobot import AbstractRobot


class BoxerRobot(AbstractRobot):
    def __init__(self):
        n = 2
        fileName = os.path.join(os.path.dirname(__file__), 'boxer.urdf')
        self._r = 0.08
        self._l = 0.494
        super().__init__(n, fileName)

    def setJointIndices(self):
        self.urdf_joints = [10, 11]
        self.robot_joints = [11, 12]
        self.castor_joints = [9, 10]

    def n(self):
        return self._n

    def reset(self, pos=None, vel=None):
        if hasattr(self, "robot"):
            p.resetSimulation()
        baseOrientation = p.getQuaternionFromEuler([0, 0, pos[2]])
        self.robot = p.loadURDF(
            fileName=self.fileName,
            basePosition=[pos[0], pos[1], 0.05],
            baseOrientation=baseOrientation,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        # Joint indices as found by p.getJointInfo()
        # set castor wheel friction to zero
        for i in self.castor_joints:
            p.setJointMotorControl2(
                self.robot,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0
            )
        # set base velocity
        v = np.zeros(2)
        v[0] = vel[0] + vel[1]
        v[1] = vel[0] - vel[0]
        self.updateState()
        self.apply_vel_action_wheels(v)
        self.state[-2:] = v
        self._vels_int = np.concatenate((self.state[-2:], self.state[13:20]))

    def setAccLimits(self):
        accLimit = np.array([1.0, 1.0])
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

    def apply_acc_action(self, accs, dt):
        self._vels_int += dt * accs
        self.apply_base_velocity(self._vels_int)

    def apply_vel_action_wheels(self, vels):
        for i in range(2):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

    def apply_base_velocity(self, vels):
        vel_left = (vels[0] - 0.5 * self._l * vels[1]) / self._r
        vel_right = (vels[0] + 0.5 * self._l * vels[1]) / self._r
        wheelVels = np.array([vel_right, vel_left])
        self.apply_vel_action_wheels(wheelVels)

    def updateState(self):
        # Get Base State
        linkState = p.getLinkState(self.robot, 0, computeLinkVelocity=1)
        posBase = np.array(
            [
                linkState[0][0],
                linkState[0][1],
                p.getEulerFromQuaternion(linkState[1])[2],
            ]
        )
        velBase = np.array([linkState[6][0], linkState[6][1], linkState[7][2]])
        vf = np.array([np.sqrt(velBase[0] ** 2 + velBase[1] ** 2), velBase[2]])

        self.state = np.concatenate((posBase, velBase, vf))

