from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ontology import Ontology


@dataclass
class EmbeddingModel(ABC):
    ontology: Ontology

    def set_ontology(self, ontology: Ontology):
        self.ontology = ontology

    @abstractmethod
    def embed_image(self, input: Any) -> np.array:
        pass

    @abstractmethod
    def embed_text(self, input: Any) -> np.array:
        pass
