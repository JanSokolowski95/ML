import numpy as np
from abc import ABC, abstractmethod


class Cost(ABC):
    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def calculate(self, y, pred): ...


class MSE(Cost):
    def __init__(self):
        super().__init__()

    def calculate(self, y, pred):
        return np.mean((y - pred) ** 2)
