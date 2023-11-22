from dataclasses import dataclass
from typing import List, Tuple
import roboflow

from autodistill.detection.detection_ontology import DetectionOntology


@dataclass
class FewShotOntology(DetectionOntology):
    promptMap: List[Tuple[str, str]]

    def __init__(self, roboflow_dataset_id: str, number_of_shots: int):
        roboflow.login()

        self.roboflow_dataset_id = roboflow_dataset_id
        self.number_of_shots = number_of_shots
        self.promptMap = []

        if len(self.promptMap) == 0:
            raise ValueError("Ontology is empty")

    def prompts(self) -> List[str]:
        return super().prompts()

    def classToPrompt(self, cls: str) -> str:
        return super().classToPrompt(cls)
