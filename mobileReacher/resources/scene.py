import pybullet as p
import pybullet_data
import os


class Scene:
    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(
            "table/table.urdf",
            basePosition=[0.8, 0.0, -0.1],
            baseOrientation=[0.0, 0.0, 0.707, 0.707],
        )
