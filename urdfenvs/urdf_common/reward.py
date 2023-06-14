from abc import ABC, abstractmethod

class Reward(ABC):
    @abstractmethod
    def calculateReward(self, observation) -> float: pass
