# pylint: disable-all
import numpy as np
from urdfpy import URDF
import os
import urdfenvs.panda_reacher

from nLinkUrdfReacher.resources.accController import AccController

class PandaAccController(AccController):

    def __init__(self):
        n = 1
        k = np.zeros(n)
        urdf_file = os.path.dirname(panda_reacher.__file__) + "/resources/pandaSmall.urdf"
        robot = URDF.load(urdf_file)
        m = np.zeros(n)
        I = np.zeros((9, n))
        off_j = np.zeros((16, n))
        off_ee = np.zeros((16, n))
        com = np.zeros((3, n))
        g = np.array([-10.0])
        axis = np.zeros((3, n))
        for i in range(n):
            link = robot.links[i + 2]
            m[i] = link.inertial.mass
            I[:, i] = link.inertial.inertia.flatten()
            com[:, i] = link.inertial.origin[:-1, -1]
            joint = robot.joints[i + 1]
            joint1 = robot.joints[i + 1]
            off_j[:, i] = np.transpose(joint.origin).flatten()
            off_ee[:, i] = np.transpose(joint1.origin).flatten()
            axis[:, i] = joint.axis


        print("m : ", m)
        print("com : ", com.transpose())
        for i in range(n):
            print("I : ", np.reshape(I[:, i], (3, 3)))
        print("axis : ", axis.transpose())
        for i in range(n):
            print("off_j : ", np.reshape(off_j[:, i], (4, 4)))
        super(PandaAccController, self).__init__(n, com, m, I, g, axis, off_j, k)
