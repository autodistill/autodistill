from dataclasses import dataclass
from typing import Any, List, Tuple
import numpy as np

from autodistill.core.ontology import Ontology


@dataclass
class EmbeddingOntology(Ontology):
    promptMap: List[Tuple[np.ndarray, str]]

    def prompts(self) -> List[np.ndarray]:
        return [prompt for prompt, _ in self.promptMap]

    def classes(self) -> List[str]:
        return [cls for _, cls in self.promptMap]
