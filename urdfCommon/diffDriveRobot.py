import pybullet as p
import gym
from urdfpy import URDF
import numpy as np

from urdfCommon.abstractRobot import AbstractRobot


class DiffDriveRobot(AbstractRobot):
    def __init__(self, n, fileName):
        super().__init__(n, fileName)

    def ns(self):
        return self.n() + 1

    def reset(self, pos=None, vel=None):
        if hasattr(self, "robot"):
            p.resetSimulation()
        baseOrientation = p.getQuaternionFromEuler([0, 0, pos[2]])
        self.robot = p.loadURDF(
            fileName=self.fileName,
            basePosition=[pos[0], pos[1], 0.15],
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
        for i in range(2, self._n):
            p.resetJointState(
                self.robot, 
                self.robot_joints[i],
                pos[i + 1],
                targetVelocity=vel[i],
            )
        # set base velocity
        self.updateState()
        self._vels_int = vel

    def readLimits(self):
        robot = URDF.load(self.fileName)
        self._limitPos_j = np.zeros((2, self.ns()))
        self._limitVel_j = np.zeros((2, self.ns()))
        self._limitTor_j = np.zeros((2, self.n()))
        self._limitAcc_j = np.zeros((2, self.n()))
        for i in range(self.n()):
            joint = robot.joints[self.urdf_joints[i]]
            print(joint.name)
            self._limitTor_j[0, i] = -joint.limit.effort
            self._limitTor_j[1, i] = joint.limit.effort
            if i >= 2:
                self._limitPos_j[0, i+1] = joint.limit.lower
                self._limitPos_j[1, i+1] = joint.limit.upper
                self._limitVel_j[0, i+1] = -joint.limit.velocity
                self._limitVel_j[1, i+1] = joint.limit.velocity
        self._limitVelForward_j = np.array([[-4, -10], [4, 10]])
        self._limitPos_j[0, 0:3] = np.array([-10, -10, -2 * np.pi])
        self._limitPos_j[1, 0:3] = np.array([10, 10, 2 * np.pi])
        self._limitVel_j[0, 0:3] = np.array([-4, -4, -10])
        self._limitVel_j[1, 0:3] = np.array([4, 4, 10])
        self.setAccLimits()

    def getLimits(self):
        return (self._limitPos_j, self._limitVel_j, self._limitTor_j)

    def getObservationSpace(self):
        return gym.spaces.Dict({
            'x': gym.spaces.Box(low=self._limitPos_j[0, :], high=self._limitPos_j[1, :], dtype=np.float64), 
            'vel': gym.spaces.Box(low=self._limitVelForward_j[0, :], high=self._limitVelForward_j[1, :], dtype=np.float64),
            'xdot': gym.spaces.Box(low=self._limitVel_j[0, :], high=self._limitVel_j[1, :], dtype=np.float64), 
        })

    def apply_torque_action(self, torques):
        for i in range(2, self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.TORQUE_CONTROL,
                                        force=torques[i])

    def apply_acc_action(self, accs, dt):
        self._vels_int += dt * accs
        self._vels_int[0] = np.clip(self._vels_int[0], 0.7 * self._limitVelForward_j[0, 0], 0.7 * self._limitVelForward_j[1, 0])
        self._vels_int[1] = np.clip(self._vels_int[1], 0.5 * self._limitVelForward_j[0, 1], 0.5 * self._limitVelForward_j[1, 1])
        self.apply_base_velocity(self._vels_int)
        self.apply_vel_action(self._vels_int)

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

    def apply_vel_action(self, vels):
        for i in range(2, self._n):
            p.setJointMotorControl2(self.robot, self.robot_joints[i],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=vels[i])

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
        # Get Joint Configurations
        joint_pos_list = []
        joint_vel_list = []
        for i in range(2, self._n):
            pos, vel, _, _ = p.getJointState(self.robot, self.robot_joints[i])
            joint_pos_list.append(pos)
            joint_vel_list.append(vel)
        joint_pos = np.array(joint_pos_list)
        joint_vel = np.array(joint_vel_list)

        # Concatenate position[0:10], velocity[0:10], vf[0:3]
        self.state = {'x': np.concatenate((posBase, joint_pos)), 'vel': vf, 'xdot': np.concatenate((velBase, joint_vel))}

    def updateSensing(self):
        self.sensor_observation = {}
        for sensor in self._sensors:
            self.sensor_observation[sensor.name()] = sensor.sense(self.robot)

    def get_observation(self):
        self.updateState()
        self.updateSensing()
        return {**self.state, **self.sensor_observation}

    def addSensor(self, sensor):
        self._sensors.append(sensor)
        return sensor.getOSpaceSize()

