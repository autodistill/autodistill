from abc import abstractmethod
from dataclasses import dataclass

from autodistill.core import BaseModel

from .text_classification_ontology import TextClassificationOntology

@dataclass
class TextClassificationBaseModel(BaseModel):
    ontology: TextClassificationOntology

    @abstractmethod
    def predict(self, input: str) -> dict:
        pass

    def label(
        self,
        input_jsonl: str,
        output_jsonl: str = "output.jsonl",
    ) -> None:
        """
        Label a dataset with the model.
        """
        pass