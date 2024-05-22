from abc import abstractmethod
from autodistill.core import TargetModel


class TextClassificationTargetModel(TargetModel):
    @abstractmethod
    def __init__(self, model_name = None):
        pass

    @abstractmethod
    def predict(self, input: str) -> dict:
        pass

    @abstractmethod
    def train(self, dataset_file, output_dir="output", epochs=5) -> None:
        pass
