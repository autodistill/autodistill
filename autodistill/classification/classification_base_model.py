import glob
import os
from abc import abstractmethod
from dataclasses import dataclass

import cv2
import supervision as sv
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.detection import CaptionOntology


@dataclass
class ClassificationBaseModel(BaseModel):
    """
    Use a foundation classification model to auto-label data.
    """

    ontology: CaptionOntology

    @abstractmethod
    def predict(self, input: str) -> sv.Classifications:
        """
        Run inference on the model.
        """
        pass

    def label(
        self, input_folder: str, extension: str = ".jpg", output_folder: str = None
    ) -> sv.ClassificationDataset:
        """
        Label a dataset and save it in a classification folder structure.
        """
        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        images_map = {}
        detections_map = {}

        files = glob.glob(input_folder + "/*" + extension)
        progress_bar = tqdm(files, desc="Labeling images")
        # iterate through images in input_folder
        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)
            image = cv2.imread(f_path)

            f_path_short = os.path.basename(f_path)
            images_map[f_path_short] = image.copy()
            detections = self.predict(f_path)
            detections_map[f_path_short] = detections

        dataset = sv.ClassificationDataset(
            self.ontology.classes(), images_map, detections_map
        )

        train_cs, test_cs = dataset.split(
            split_ratio=0.7, random_state=None, shuffle=True
        )
        test_cs, valid_cs = test_cs.split(
            split_ratio=0.5, random_state=None, shuffle=True
        )

        train_cs.as_folder_structure(root_directory_path=output_folder + "/train")

        test_cs.as_folder_structure(root_directory_path=output_folder + "/test")

        valid_cs.as_folder_structure(root_directory_path=output_folder + "/valid")

        print("Labeled dataset created - ready for distillation.")
        return dataset
