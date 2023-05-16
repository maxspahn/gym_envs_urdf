from abc import ABC, abstractmethod
from typing import List
from urdfenvs.sensors.sensor import Sensor


class Reward(ABC):
    @abstractmethod
    # TODO: Get a clear list of things that could be needed for calculating reward.
    def calculateReward(self, observation) -> float: pass

# class Reward1(Reward):
#     def calculateReward(self, sensors: list[Sensor]) -> float:
#         return 1
