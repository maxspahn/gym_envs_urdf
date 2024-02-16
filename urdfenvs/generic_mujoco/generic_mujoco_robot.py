import os
from abc import abstractmethod
from typing import Dict
import pybullet as p
import gymnasium as gym
import numpy as np
import urdfenvs
from urdfenvs.sensors.sensor import Sensor

class GenericMujocoRobot():
    """GenericMujocoRobot."""

    _xml_file: str

    def __init__(self, xml_file: str, mode: str = "vel"):
        """Constructor for generic robot.

        Parameters
        ----------

        xml_file: str :
            Name of xml file.
        mode: str:
            Control mode. Note that the mode is not used in mujoco as it is implicitely defined by the actuators.
        """
        if not os.path.exists(xml_file):
            raise Exception(f"the request xml {xml_file} can not be found")
        else:
            self._xml_file = xml_file

    @property
    def xml_file(self) -> str:
        return self._xml_file

