import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import supervision as sv

from .ontology import Ontology


@dataclass
class BaseModel(ABC):
    ontology: Ontology

    def __init__(self, ontology: Ontology):
        super().__init__()

    def set_ontology(self, ontology: Ontology):
        self.ontology = ontology

    @abstractmethod
    def predict(self, input: Any) -> Any:
        pass

    @abstractmethod
    def label(
        self, input_folder: str, extension: str = ".jpg", output_folder: str = None
    ) -> sv.BaseDataset:
        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        config = {"start_time": time.time(), "base_model": self.__class__.__name__}

        with open(os.path.join(output_folder, "config.json"), "w+") as f:
            json.dump(config, f)

        return output_folder, config
