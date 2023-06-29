from abc import ABC, abstractmethod

from autodistill.core import TargetModel
import supervision as sv

class ClassificationTargetModel(TargetModel):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, input:str, confidence:float = 0.5) -> sv.Classifications:
        pass

    @abstractmethod
    def train(self):
        pass
