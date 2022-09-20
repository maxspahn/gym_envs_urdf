from os import stat
import numpy as np
from gym import core, spaces
from numpy.core.defchararray import array
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation


def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes

    Returns
        quat_dot, [i,j,k,w]

    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Augment to maintain unit quaternion.
    quat_err = np.sum(quat**2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot


class Quadrotor(core.Env):
    """
    Quadrotor forward dynamics model.
    """

    def __init__(self):
        self.mass            = 0.030  # kg
        self.Ixx             = 1.43e-5  # kg*m^2
        self.Iyy             = 1.43e-5  # kg*m^2
        self.Izz             = 2.89e-5  # kg*m^2
        self.arm_length      = 0.046  # meters
        self.rotor_speed_min = 0  # rad/s
        self.rotor_speed_max = 2500  # rad/s
        self.k_thrust        = 2.3e-08  # N/(rad/s)**2
        self.k_drag          = 7.8e-11   # Nm/(rad/s)**2

        # Additional constants.
        self.inertia = np.diag(
            np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2

        # Precomputes
        k = self.k_drag/self.k_thrust
        L = self.arm_length
        self.to_TM = np.array([[1,  1,  1,  1],
                               [0,  L,  0, -L],
                               [-L,  0,  L,  0],
                               [k, -k,  k, -k]])
        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass*self.g])
        self.t_step = 0.01

    def reset(self, position=[0, 0, 0], yaw=0, pitch=0, roll=0):
        '''
        state is a 13 dimensional vector
            postion*3 velocity*3 attitude(quaternion)*4 angular velocity*3
        state = [x y z dx dy dz qx qy qz qw r p q]
        dot_state = [dx dy dz ddx ddy ddz dqx dqy dqz dqw dr dp dq]
        '''
        quaternion = Rotation.from_euler(
            'zyx', [yaw, pitch, roll], degrees=True).as_quat()
        s = np.zeros(13)
        s[0] = position[0]
        s[1] = position[1]
        s[2] = position[2]
        s[3] = 0
        s[4] = 0
        s[5] = 0
        s[6] = quaternion[0]
        s[7] = quaternion[1]
        s[8] = quaternion[2]
        s[9] = quaternion[3]
        self.state = self._unpack_state(s)
        return self.state

    def step(self, cmd_rotor_speeds):
        '''
        Considering the max and min of rotor speeds
        action is a 4 dimensional vector: command rotor speeds
        action = [w1, w2, w3, w4]
        '''
        rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)
        rotor_thrusts = self.k_thrust * np.square(rotor_speeds)

        '''
        Next, [w1, w2, w3, w4] into [F Mx My Mz]
        '''
        U = self.to_TM @ rotor_thrusts
        T = U[0]
        M = U[1:]

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, T, M)

        '''
        The next state can be obtained through integration （Runge-Kutta）
        '''
        s = Quadrotor._pack_state(self.state)
        sol = scipy.integrate.solve_ivp(
            s_dot_fn, (0, self.t_step), s, first_step=self.t_step)
        s = sol['y'][:, -1]
        # turn state back to dict
        self.state = Quadrotor._unpack_state(s)

        # Re-normalize unit quaternion.
        reward = 0
        done = 0
        info = {}

        return self.state, reward, done, info

    def _s_dot_fn(self, t, s, u1, u2):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        state = Quadrotor._unpack_state(s)
        # page 73
        # Position derivative.
        x_dot = state['v']

        # Velocity derivative.
        v_dot = (self.weight + u1 * Quadrotor.rotate_k(state['q'])) / self.mass

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])

        # Angular velocity derivative. page 26 Equation 4
        w_dot = self.inv_inertia @ (u2 - Quadrotor.hat_map(
            state['w']) @ self.inertia @ state['w'])

        # Pack into vector of derivatives.
        s_dot = np.zeros((13,))
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot

        return s_dot

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([2 * (q[0] * q[2] + q[1] * q[3]),
                         2 * (q[1] * q[2] - q[0] * q[3]),
                         1 - 2 * (q[0] ** 2 + q[1] ** 2)])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        """
        return np.array([[0, -s[2], s[1]],
                         [s[2], 0, -s[0]],
                         [-s[1], s[0], 0]])

    @classmethod
    def _pack_state(cls, state):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = np.zeros((13,))
        s[0:3] = state['x']
        s[3:6] = state['v']
        s[6:10] = state['q']
        s[10:13] = state['w']
        return s

    @classmethod
    def _unpack_state(cls, s):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        """
        state = {'x': s[0:3], 'v': s[3:6], 'q': s[6:10], 'w': s[10:13]}
        return state
