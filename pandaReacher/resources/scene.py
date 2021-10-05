import pybullet as p
import pybullet_data
import os


class Scene:
    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        """
        p.loadURDF(
            "sphere_small.urdf",
            basePosition=[0.65, 0.3, 0.65],
            baseOrientation=[0.0, 0.0, 0.707, 0.707],
        )
        p.loadURDF(
            "table/table.urdf",
            basePosition=[0.7, 0.0, -0.1],
            baseOrientation=[0.0, 0.0, 0.707, 0.707],
        )
        p.loadURDF(
            "table/table.urdf",
            basePosition=[1.3, 0.2, 0.3],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.5, 0.0, 0.65],
            baseOrientation=[0.0, 0.0, 0.707, 0.707],
        )
        """
