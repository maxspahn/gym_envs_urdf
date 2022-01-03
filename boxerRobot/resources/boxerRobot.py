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
        self.urdf_joints = [2, 3]
        self.robot_joints = [3, 4]
        self.castor_joints = [1, 2]

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
        # print(self.getIndexedJointInfo())
        for i in self.castor_joints:
            p.setJointMotorControl2(
                self.robot,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0
            )
        # set base velocity
        self.updateState()
        self._vels_int = vel

    def setAccLimits(self):
        accLimit = np.array([1.0, 1.0])
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

    def apply_acc_action(self, accs, dt):
        self._vels_int += dt * accs
        self._vels_int = np.clip(self._vels_int, -np.ones(2), np.ones(2))
        #self._vels_int = self.state[3:5]
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
        linkState = p.getLinkState(self.robot, 0, computeLinkVelocity=0)
        posBase = np.array(
            [
                linkState[0][0],
                linkState[0][1],
                p.getEulerFromQuaternion(linkState[1])[2],
            ]
        )
        velWheels = p.getJointStates(self.robot, self.robot_joints)
        v_right = velWheels[0][1]
        v_left = velWheels[1][1]
        vf = np.array([0.5 * (v_right + v_left) * self._r, (v_right - v_left) * self._r / self._l])
        Jnh = np.array([[np.cos(posBase[2]), 0], [np.sin(posBase[2]), 0], [0, 1]])
        velBase = np.dot(Jnh, vf)

        self.state = np.concatenate((posBase, vf, velBase))

