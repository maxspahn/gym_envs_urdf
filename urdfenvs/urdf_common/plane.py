import pybullet as p
import os


class Plane:
    def __init__(self):
        f_name = os.path.join(os.path.dirname(__file__), 'plane.urdf')
        p.loadURDF(f_name)


