from dataclasses import dataclass
from typing import Any, List, Tuple

from autodistill.core.ontology import Ontology


@dataclass
class DetectionOntology(Ontology):
    promptMap: List[Tuple[Any, str]]

    def prompts(self) -> List[Any]:
        return [prompt for prompt, _ in self.promptMap]

    def classes(self) -> List[str]:
        return [cls for _, cls in self.promptMap]

    def promptToClass(self, prompt: Any) -> str:
        for p, cls in self.promptMap:
            if p == prompt:
                return cls
        raise ValueError("Prompt not found in ontology")

    def classToPrompt(self, cls: str) -> Any:
        for p, c in self.promptMap:
            if c == cls:
                return p
        raise ValueError("Class not found in ontology")
