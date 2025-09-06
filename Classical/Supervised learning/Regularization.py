import numpy as np
from abc import ABC, abstractmethod


class Regularization(ABC):
    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def calculate(self, y, pred): ...


class L1(Regularization):
    def __init__(self):
        super().__init__()

    def calculate(self, y, pred):
        return super().calculate(y, pred)


class L2(Regularization):
    def __init__(self):
        super().__init__()

    def calculate(self, y, pred):
        return super().calculate(y, pred)
