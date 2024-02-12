from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import supervision as sv

from autodistill.core.embedding_model import EmbeddingModel
from autodistill.core.ontology import Ontology

ONTOLOGY_WITH_EMBEDDINGS = [
    "EmbeddingOntologyRaw",
    "EmbeddingOntologyImage",
    "EmbeddingOntologyText",
]


def compare_embeddings(
    image_embedding: np.array,
    comparison_embeddings: List[np.array],
    distance_metric="cosine",
):
    """
    Calculate the similarity between an image embedding and all embeddings in a list.

    Args:
        image_embedding: The embedding of the image to compare.
        comparison_embeddings: A list of embeddings to compare against.
        distance_metric: The distance metric to use. Currently only supports "cosine".

    Returns:
        A list of similarity scores.
    """
    if distance_metric == "cosine":
        comparisons = []

        for comparison_embedding in comparison_embeddings:
            comparisons.append(
                cosine_similarity(
                    image_embedding.reshape(1, -1), comparison_embedding.reshape(1, -1)
                ).flatten()
            )

        return sv.Classifications(
            class_id=np.array([i for i in range(len(comparisons))]),
            confidence=np.array(comparisons).flatten(),
        )
    else:
        raise NotImplementedError(
            f"Distance metric {distance_metric} is not supported."
        )


@dataclass
class EmbeddingOntology(Ontology):
    embeddingMap: Dict[str, np.ndarray]

    def __init__(self, embeddingMap, cluster=1):
        self.embeddingMap = embeddingMap

    @classmethod
    def process(self, model: EmbeddingModel):
        pass

    def prompts(self) -> List[np.ndarray]:
        return [prompt for prompt, _ in self.embeddingMap]

    def classes(self) -> List[str]:
        return [cls for _, cls in self.embeddingMap]


@dataclass
class EmbeddingOntologyRaw(EmbeddingOntology):
    embeddingMap: Dict[str, np.ndarray]

    def __init__(self, embeddingMap, cluster=1):
        self.embeddingMap = embeddingMap

    def prompts(self) -> List[np.ndarray]:
        return [prompt for prompt, _ in self.embeddingMap]

    def classes(self) -> List[str]:
        return [cls for _, cls in self.embeddingMap]


@dataclass
class EmbeddingOntologyImage(EmbeddingOntology):
    # TODO: Support more than just file names
    embeddingMap: Dict[str, List]
    cluster: int

    def __init__(self, embeddingMap, cluster=1):
        self.embeddingMap = embeddingMap
        self.cluster = cluster

        if self.cluster != 1:
            print("Note: The `cluster` parameter is not currently implemented.")

    def process(self, model):
        results = {}

        for prompt, cls in self.embeddingMap.items():
            result = []

            for item in cls:
                result.append(model.embed_image(item))

            # get average of all vectors
            result = np.mean(result, axis=0)

            results[prompt] = result

        self.embeddingMap = results

    def prompts(self) -> List[np.ndarray]:
        return [prompt for prompt, _ in self.embeddingMap]

    def classes(self) -> List[str]:
        return [cls for _, cls in self.embeddingMap]


@dataclass
class EmbeddingOntologyText(EmbeddingOntology):
    embeddingMap: Dict[str, str]
    cluster: int

    def __init__(self, embeddingMap, model):
        self.embeddingMap = embeddingMap

        results = {}

        self.embeddingMap = [
            (prompt, model.embed_text(cls)) for prompt, cls in self.embeddingMap
        ]

    def prompts(self) -> List[np.ndarray]:
        return [prompt for prompt, _ in self.embeddingMap]

    def classes(self) -> List[str]:
        return [cls for _, cls in self.embeddingMap]
