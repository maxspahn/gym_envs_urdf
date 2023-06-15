from abc import ABC, abstractmethod

class Reward(ABC):
    @abstractmethod
    def calculate_reward(self, observation) -> float: pass
