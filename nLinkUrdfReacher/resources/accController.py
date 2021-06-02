import os
import numpy as np
from urdfpy import URDF

from nLinkUrdfReacher.resources.create3DDynamics import create3DDynamics
import nLinkUrdfReacher


class AccController(object):

    """Simple forward controller acceleration to torque"""

    def __init__(self, n, com, m, I, g, axis, off_j, k):
        self.dynamics_fun, _, self.tau_fun, self.M_fun = create3DDynamics(n)
        self._n = n
        self._com = com
        self._m = m
        self._I = I
        self._g = g
        self._axis = axis
        self._off_j = off_j
        self._k = k

    def control(self, q, qdot, qddot):
        t = self.M_fun(q, qdot, self._com, self._m, self._I, self._g, self._axis, self._off_j)
        tau = self.tau_fun(
            q,
            qdot,
            self._com,
            self._m,
            self._I,
            self._g,
            self._axis,
            self._off_j,
            self._k, 
            qddot,
        )
        M = t[0]
        T = t[1]
        V = t[2]
        r = t[3]
        return tau

class NLinkUrdfAccController(AccController):

    def __init__(self, n, k):
        urdf_file = os.path.dirname(nLinkUrdfReacher.__file__) + "/resources/nlink_" + str(n) + ".urdf"
        robot = URDF.load(urdf_file)
        m = np.zeros(n)
        I = np.zeros((9, n))
        off_j = np.zeros((16, n))
        off_ee = np.zeros((16, n))
        com = np.zeros((3, n))
        g = np.array([-10.000])
        axis = np.zeros((3, n))
        for i in range(n):
            link = robot.links[i + 2]
            m[i] = link.inertial.mass
            I[:, i] = link.inertial.inertia.flatten()
            com[:, i] = link.inertial.origin[:-1, -1]
            joint = robot.joints[i + 1]
            joint1 = robot.joints[i + 2]
            off_j[:, i] = np.transpose(joint.origin).flatten()
            off_ee[:, i] = np.transpose(joint1.origin).flatten()
            axis[:, i] = joint.axis

        print("com : ", com)
        print("m : ", m)
        print("I : ", I)
        print("off_j : ", off_j)
        print("axis : ", axis)


        super(NLinkUrdfAccController, self).__init__(n, com, m, I, g, axis, off_j, k)
