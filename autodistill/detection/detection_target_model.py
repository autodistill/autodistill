from abc import ABC, abstractmethod

import supervision as sv

from autodistill.core import TargetModel


class DetectionTargetModel(TargetModel):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, input: str, confidence: float = 0.5) -> sv.Detections:
        pass

    @abstractmethod
    def train(self):
        pass
