from dataclasses import dataclass
from typing import Any, List

import numpy as np

from autodistill.core.ontology import Ontology


@dataclass
class EmbeddingOntology(Ontology):
    promptMap: List[str, Any]
    embeddingMap: List[str, np.ndarray]
    cluster: int

    def __init__(self, embeddingMap, cluster=1):
        self.embeddingMap = embeddingMap
        self.cluster = cluster

        if self.cluster != 1:
            print("The `cluster` parameter is not currently implemented.")

    def process(self, inference_callback):
        if self.embeddingMap:
            return

        for prompt, cls in self.promptMap:
            result = []

            for item in cls:
                # inference callback should support image paths or np arrays
                result.append(inference_callback(item))

            # get average of all vectors
            result = np.mean(result, axis=0)

            self.embeddingMap[prompt] = result

    def prompts(self) -> List[np.ndarray]:
        return [prompt for prompt, _ in self.promptMap]

    def classes(self) -> List[str]:
        return [cls for _, cls in self.promptMap]
