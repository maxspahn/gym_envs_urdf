import pybullet as p
import pybullet_data
import gym
import os
import math
from urdfpy import URDF
import numpy as np
import urdf2casadi.urdfparser as u2c


class PandaRobot:
    def __init__(self, gripper=False):
        self._gripper = gripper
        if gripper:
            self.f_name = os.path.join(os.path.dirname(__file__), 'pandaWithGripper_working.urdf')
            self._n = 9
            # bullet meshes
            #self.robot_joints = [0, 1, 2, 3, 4, 5, 6, 9, 10]
            #self.control_joints = [0, 1, 2, 3, 4, 5, 6, 9, 10]
            # culstom meshes
            self.robot_joints = [0, 1, 2, 3, 4, 5, 6, 8, 9]
            self.control_joints = [0, 1, 2, 3, 4, 5, 6, 8, 9]
            self.getId(tip="panda_link7")
        else:
            self.f_name = os.path.join(os.path.dirname(__file__), 'panda_working.urdf')
            self._n = 7
            self.robot_joints = [1, 2, 3, 4, 5, 6, 7]
            self.control_joints = [1, 2, 3, 4, 5, 6, 7]
            self.getId(tip="panda_link7")
        self.readLimits()

    def getId(self, tip):
        """
        rbdl_file = self.f_name[:-5] + '_no_world.urdf'
        self._panda_rbdl = rbdl.loadModel(rbdl_file)
        """
        panda_u2c = u2c.URDFparser()
        panda_u2c.from_file(self.f_name)
        root = 'panda_link0'
        self._id_u2c = panda_u2c.get_inverse_dynamics_rnea(root, tip, [0, 0, -9.81])

    def addObstacle(self, pos, filename):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(
            filename,
            basePosition=pos
        )

    def reset(self, poss=np.array([0.0, 0.0, 0.0, -1.501, 0.0, 1.8675, 0.0, 0.02, 0.02])):
        self.robot = p.loadURDF(fileName=self.f_name, useFixedBase=True,
                            flags=p.URDF_USE_INERTIA_FROM_FILE, 
                              basePosition=[0, 0, 0.2])
        # Joint indices as found by p.getJointInfo()
        numJoints = p.getNumJoints(self.robot)
        for i in range(self._n):
            p.setJointMotorControl2(self.robot, self.control_joints[i],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=poss[i])
        print("Bringing to initial position..")
        pre_steps = 1000
        for i in range(pre_steps):
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
        self._limitAcc_j[0, :] = -accLimit[0:self._n]
        self._limitAcc_j[1, :] = accLimit[0:self._n]

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

    def disableVelocityControl(self):
        self._friction = 0
        for i in range(self._n):
            p.setJointMotorControl2(
                self.robot,
                jointIndex=self.robot_joints[i],
                controlMode=p.VELOCITY_CONTROL,
                force=self._friction
            )

    def get_ids(self):
        return self.robot

    def apply_torque_action(self, torques):
        for i in range(self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.TORQUE_CONTROL,
                                        force=torques[i])

    def apply_acc_action(self, accs):
        for i in range(len(accs)):
            accs[i] = np.clip(accs[i], self._limitAcc_j[0, i], self._limitAcc_j[1, i])
        q = []
        qdot = []
        for i in range(self._n):
            pos, vel, _, _= p.getJointState(self.robot, self.control_joints[i])
            q.append(pos)
            qdot.append(vel)
        """
        tau_rbdl = np.zeros(self._n)
        rbdl.InverseDynamics(self._panda_rbdl, np.array(q), np.array(qdot), accs, tau_rbdl)
        """
        qddot = list(accs)
        q = list(q)
        qdot = list(qdot)
        tau = p.calculateInverseDynamics(self.robot, q, qdot, qddot)
        #print("----")
        #tau = self._id_u2c(q, qdot, qddot)
        #print("tau_rbdl : ", tau_rbdl)
        #print("tau_pb : ", np.array(tau))
        #print(tau_rbdl - np.array(tau))
        #print(np.linalg.norm(tau_rbdl - np.array(tau)))
        #print("----")
        self.apply_torque_action(tau)

    def apply_vel_action(self, vels):
        for i in range(self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

    def get_observation(self):
        # Get Joint Configurations
        joint_pos_list = []
        joint_vel_list = []
        for i in range(self._n):
            pos, vel, _, _= p.getJointState(self.robot, self.control_joints[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = tuple(joint_pos_list)
        joint_vel = tuple(joint_vel_list)

        # Concatenate position, orientation, velocity
        self.observation = (joint_pos+ joint_vel)
        return self.observation

if __name__ == "__main__":
    robot = PandaRobot(gripper=False)
