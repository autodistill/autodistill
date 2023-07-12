from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import supervision as sv

from autodistill.core import Ontology


@dataclass
class BaseModel(ABC):
    ontology: Ontology

    def set_ontology(self, ontology: Ontology):
        self.ontology = ontology

    @abstractmethod
    def predict(self, input: Any) -> Any:
        pass

    @abstractmethod
    def label(
        self, input_folder: str, extension: str = ".jpg", output_folder: str = None
    ) -> sv.BaseDataset:
        pass
