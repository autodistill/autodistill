import glob
import os
from abc import abstractmethod
from dataclasses import dataclass
import datetime

import cv2
import supervision as sv
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.detection import CaptionOntology
from autodistill.helpers import split_data


@dataclass
class ClassificationBaseModel(BaseModel):
    ontology: CaptionOntology

    @abstractmethod
    def predict(self, input: str) -> sv.Classifications:
        pass

    def label(
        self, input_folder: str, extension: str = ".jpg", output_folder: str = None
    ) -> sv.ClassificationDataset:
        images_map = {}
        detections_map = {}

        # if output_folder/autodistill.json exists
        if os.path.exists(output_folder + "/data.yaml"):
            dataset = sv.ClassificationDataset.from_yolo(
                output_folder + "/images",
                output_folder + "/annotations",
                output_folder + "/data.yaml",
            )

            # DetectionsDataset iterator returns
            # image_name, image, self.annotations.get(image_name, None)
            # ref: https://supervision.roboflow.com/datasets/#supervision.dataset.core.DetectionDataset
            for item in dataset:
                image_name = item[0]
                image = item[1]
                detections = item[2]

                image_base_name = os.path.basename(image_name)

                images_map[image_base_name] = image
                detections_map[image_base_name] = detections

        files = glob.glob(input_folder + "/*" + extension)
        progress_bar = tqdm(files, desc="Labeling images")
        # iterate through images in input_folder
        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)
            image = cv2.imread(f_path)

            f_path_short = os.path.basename(f_path)
            images_map[f_path_short] = image.copy()

            annotation_path = os.path.join(output_folder, "annotations/", ".".join(f_path_short.split(".")[:-1]) + ".txt")

            if not os.path.exists(annotation_path):
                detections = self.predict(f_path)
                detections_map[f_path_short] = detections

        dataset = sv.ClassificationDataset(
            self.ontology.classes(), images_map, detections_map
        )

        train_cs, test_cs = dataset.split(split_ratio=0.7)
        test_cs, valid_cs = test_cs.split(split_ratio=0.5)

        train_cs.as_folder_structure(root_directory_path=output_folder + "/train")

        test_cs.as_folder_structure(root_directory_path=output_folder + "/test")

        valid_cs.as_folder_structure(root_directory_path=output_folder + "/valid")

        print("Labeled dataset created - ready for distillation.")
        return dataset, output_folder
