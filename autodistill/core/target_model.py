from abc import ABC, abstractmethod


class TargetModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(sel, input):
        pass

    @abstractmethod
    def train(self):
        pass
