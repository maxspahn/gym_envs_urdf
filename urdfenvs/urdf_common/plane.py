import os
import pybullet


class Plane:
    def __init__(self):
        f_name = os.path.join(os.path.dirname(__file__), 'plane.urdf')
        pybullet.loadURDF(f_name)


