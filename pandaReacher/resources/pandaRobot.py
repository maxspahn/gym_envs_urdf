import pybullet as p
import pybullet_data
import gym
import os
from urdfpy import URDF
import numpy as np


class PandaRobot:
    def __init__(self, gripper=False, friction=0.0):
        self._gripper = gripper
        self._friction = friction
        if gripper:
            self.f_name = os.path.join(
                os.path.dirname(__file__), "pandaWithGripper_working.urdf"
            )
            self._n = 9
            self.robot_joints = [0, 1, 2, 3, 4, 5, 6, 8, 9]
            self.control_joints = [0, 1, 2, 3, 4, 5, 6, 8, 9]
        else:
            self.f_name = os.path.join(os.path.dirname(__file__), "panda_working.urdf")
            self._n = 7
            self.robot_joints = [1, 2, 3, 4, 5, 6, 7]
            self.control_joints = [1, 2, 3, 4, 5, 6, 7]
        self.readLimits()

    def addObstacle(self, pos, filename):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(filename, basePosition=pos)

    def reset(
        self, poss=np.array([0.0, 0.0, 0.0, -1.501, 0.0, 1.8675, 0.0, 0.02, 0.02])
    ):
        self.robot = p.loadURDF(
            fileName=self.f_name,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            basePosition=[0, 0, 0.0],
        )
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                self.control_joints[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=poss[i],
            )
        print("Bringing to initial position..")
        pre_steps = 1000
        for i in range(pre_steps):
            if i % 100 == 0:
                print("Completed %d of %d steps to inital position" % (i, pre_steps))
            p.stepSimulation()
        print("Reached initial position")

    def getLimits(self):
        return (self._limitPos_j, self._limitVel_j, self._limitTor_j, self._limitAcc_j)

    def readLimits(self):
        robot = URDF.load(self.f_name)
        self._limitPos_j = np.zeros((2, self._n))
        self._limitVel_j = np.zeros((2, self._n))
        self._limitAcc_j = np.zeros((2, self._n))
        self._limitTor_j = np.zeros((2, self._n))
        for i, j in enumerate(self.control_joints):
            joint = robot.joints[j]
            print("joint : ", joint.name)
            self._limitPos_j[0, i] = joint.limit.lower
            self._limitPos_j[1, i] = joint.limit.upper
            self._limitVel_j[0, i] = -joint.limit.velocity
            self._limitVel_j[1, i] = joint.limit.velocity
            self._limitTor_j[0, i] = -joint.limit.effort
            self._limitTor_j[1, i] = joint.limit.effort
        accLimit = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0, 1.0, 1.0])
        self._limitAcc_j[0, :] = -accLimit[0: self._n]
        self._limitAcc_j[1, :] = accLimit[0: self._n]

    def getTorqueSpaces(self):
        xu = np.concatenate((self._limitPos_j[1, :], self._limitVel_j[1, :]))
        xl = np.concatenate((self._limitPos_j[0, :], self._limitVel_j[0, :]))
        uu = self._limitTor_j[1, :]
        ul = self._limitTor_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def getAccSpaces(self):
        xu = np.concatenate((self._limitPos_j[1, :], self._limitVel_j[1, :]))
        xl = np.concatenate((self._limitPos_j[0, :], self._limitVel_j[0, :]))
        uu = self._limitAcc_j[1, :]
        ul = self._limitAcc_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def getVelSpaces(self):
        xu = self._limitPos_j[1, :]
        xl = self._limitPos_j[0, :]
        uu = self._limitVel_j[1, :]
        ul = self._limitVel_j[0, :]
        ospace = gym.spaces.Box(low=xl, high=xu, dtype=np.float64)
        aspace = gym.spaces.Box(low=ul, high=uu, dtype=np.float64)
        return (ospace, aspace)

    def disableVelocityControl(self):
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                jointIndex=self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                force=self._friction,
            )

    def get_ids(self):
        return self.robot

    def apply_torque_action(self, torques):
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.TORQUE_CONTROL,
                force=torques[i],
            )

    def apply_acc_action(self, accs):
        for i in range(len(accs)):
            accs[i] = np.clip(accs[i], self._limitAcc_j[0, i], self._limitAcc_j[1, i])
        q = []
        qdot = []
        for i in range(self._n):
            pos, vel, _, _ = p.getJointState(self.robot, self.control_joints[i])
            q.append(pos)
            qdot.append(vel)
        qddot = list(accs)
        q = list(q)
        qdot = list(qdot)
        tau = p.calculateInverseDynamics(self.robot, q, qdot, qddot)
        self.apply_torque_action(tau)

    def apply_vel_action(self, vels):
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vels[i],
            )

    def get_observation(self):
        # Get Joint Configurations
        joint_pos_list = []
        joint_vel_list = []
        for i in range(self._n):
            pos, vel, _, _ = p.getJointState(self.robot, self.control_joints[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = np.array(joint_pos_list)
        joint_vel = np.array(joint_vel_list)

        # Concatenate position, orientation, velocity
        return np.concatenate((joint_pos, joint_vel))
