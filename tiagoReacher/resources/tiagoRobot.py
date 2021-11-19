import pybullet as p
import gym
import os
import math
from urdfpy import URDF
import numpy as np


class TiagoRobot:
    def __init__(self):
        self._n = 19
        self.f_name = os.path.join(os.path.dirname(__file__), 'tiago_dual.urdf')
        self.setJointIdsUrdf()
        self.readLimits()

    def setJointIdsUrdf(self):
        # Finding out the joint names
        wheel_joint_names = ['wheel_right_joint', 'wheel_left_joint']
        torso_joint_name = ['torso_lift_joint']
        head_joint_names = ['head_' + str(i) + '_joint' for i in range(3)]
        arm_right_joint_names = ['arm_right_' + str(i) + '_joint' for i in range(8)]
        arm_left_joint_names = ['arm_left_' + str(i) + '_joint' for i in range(8)]
        self._joint_names = wheel_joint_names + torso_joint_name + head_joint_names + arm_right_joint_names + arm_left_joint_names
        robot = URDF.load(self.f_name)
        self.robot_joints_urdf = []
        for i, joint in enumerate(robot.joints):
            if joint.name in self._joint_names:
                self.robot_joints_urdf.append(i)
        return

    def setJointIdsControl(self):
        self.robot_joints_control = []
        self.caster_joints = []
        for _id in range(p.getNumJoints(self.robot)):
            joint_name = p.getJointInfo(self.robot, _id)[1].decode('UTF-8')
            if joint_name in self._joint_names:
                self.robot_joints_control.append(_id)
            if 'caster' in joint_name:
                self.caster_joints.append(_id)
        self.robot_joints_gripper = []

    def reset(self):
        self.robot = p.loadURDF(
                        fileName=self.f_name,
                        basePosition=[0, 0, 0.15], 
                        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self.setJointIdsControl()
        # Joint indices as found by p.getJointInfo()
        # set castor wheel friction to zero
        for i in self.caster_joints:
            p.setJointMotorControl2(
                self.robot,
                jointIndex=i,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0
            )

    def getLimits(self):
        return (self._limitPos_j, self._limitVel_j, self._limitTor_j, self._limitAcc_j)

    def readLimits(self):
        robot = URDF.load(self.f_name)
        self._limitPos_j = np.zeros((2, self._n))
        self._limitVel_j = np.zeros((2, self._n))
        self._limitAcc_j = np.zeros((2, self._n))
        self._limitTor_j = np.zeros((2, self._n))
        for i, j in enumerate(self.robot_joints_urdf):
            joint = robot.joints[j]
            self._limitPos_j[0, i] = joint.limit.lower
            self._limitPos_j[1, i] = joint.limit.upper
            self._limitVel_j[0, i] = -joint.limit.velocity
            self._limitVel_j[1, i] = joint.limit.velocity
            self._limitTor_j[0, i] = -joint.limit.effort
            self._limitTor_j[1, i] = joint.limit.effort
        # manually done
        accLimit = np.ones(self._n)
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

    def getTorqueSpaces(self):
        xu = np.concatenate((self._limitPos_j[1, :], self._limitVel_j[1, :]))
        xl = np.concatenate((self._limitPos_j[0, :], self._limitVel_j[0, :]))
        uu = self._limitTor_j[1, :]
        ul = self._limitTor_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return(ospace, aspace)

    def getAccSpaces(self):
        xu = np.concatenate((self._limitPos_j[1, :], self._limitVel_j[1, :]))
        xl = np.concatenate((self._limitPos_j[0, :], self._limitVel_j[0, :]))
        uu = self._limitAcc_j[1, :]
        ul = self._limitAcc_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return(ospace, aspace)

    def getVelSpaces(self):
        xu = self._limitPos_j[1, :]
        xl = self._limitPos_j[0, :]
        uu = self._limitVel_j[1, :]
        ul = self._limitVel_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return(ospace, aspace)

    def disableVelocityControl(self, friction):
        for i in range(2, self._n):
            p.setJointMotorControl2(
                self.robot,
                jointIndex=self.robot_joints_control[i],
                controlMode=p.VELOCITY_CONTROL,
                force=friction
            )

    def get_ids(self):
        return self.robot

    def apply_torque_action(self, torques):
        for i in range(2, self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints_control[i],
                                        controlMode=p.TORQUE_CONTROL,
                                        force=torques[i])

    def apply_acc_action(self, accs):
        q = []
        qdot = []
        for i in range(2, self._n):
            pos, vel, _, _= p.getJointState(self.robot, self.robot_joints_control[i])
            q.append(pos)
            qdot.append(vel)
        qddot = list(accs)
        q = list(q)
        qdot = list(qdot)
        # TODO: Fails probable due to mismatch in dimension <29-09-21, mspahn> #
        tau = p.calculateInverseDynamics(self.robot, q, qdot, qddot)
        self.apply_torque_action(tau)

    def apply_base_velocity(self, vels):
        r = 0.1
        wheelVels = vels/r
        self.apply_vel_action_wheels(wheelVels)

    def apply_vel_action_wheels(self, vels):
        for i in range(2):
            p.setJointMotorControl2(self.robot, self.robot_joints_control[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

    def apply_vel_action(self, vels):
        for i in range(2, self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints_control[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

    def get_observation(self):
        # Get Base State
        linkState = p.getLinkState(self.robot, 0, computeLinkVelocity=1)
        posBase = np.array([linkState[0][0], linkState[0][1], p.getEulerFromQuaternion(linkState[1])[2]])
        velBase = np.array([linkState[6][0], linkState[6][1], linkState[7][2]])
        # Get Joint Configurations
        joint_pos_list = []
        joint_vel_list = []
        for i in range(2, self._n):
            pos, vel, _, _ = p.getJointState(self.robot, self.robot_joints_control[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = np.array(joint_pos_list)
        joint_vel = np.array(joint_vel_list)

        # Concatenate position, orientation, velocity
        self.observation = np.concatenate((posBase, joint_pos, velBase, joint_vel))
        return self.observation

if __name__ == "__main__":
    mr = TiagoRobot()
