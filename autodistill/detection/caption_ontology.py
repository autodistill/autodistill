from dataclasses import dataclass
from typing import Dict, List, Tuple

from autodistill.detection.detection_ontology import DetectionOntology


@dataclass
class CaptionOntology(DetectionOntology):
    promptMap: List[Tuple[str, str]]

    def __init__(self, ontology: Dict[str, str]):
        self.promptMap = [(k, v) for k, v in ontology.items()]

        if len(self.promptMap) == 0:
            raise ValueError("Ontology is empty")

    def prompts(self) -> List[str]:
        return super().prompts()

    def classToPrompt(self, cls: str) -> str:
        return super().classToPrompt(cls)
