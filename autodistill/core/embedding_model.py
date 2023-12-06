from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ontology import Ontology


@dataclass
class EmbeddingModel(ABC):
    """
    Use an embedding model to calculate embeddings for use in classification.
    """

    ontology: Ontology

    def set_ontology(self, ontology: Ontology):
        """
        Set the ontology for the model.
        """
        self.ontology = ontology

    @abstractmethod
    def embed_image(self, input: Any) -> np.array:
        """
        Calculate an image embedding for an image.
        """
        pass

    @abstractmethod
    def embed_text(self, input: Any) -> np.array:
        """
        Calculate a text embedding for an image.
        """
        pass
