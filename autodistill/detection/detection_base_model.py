import glob
import os
from abc import abstractmethod
from dataclasses import dataclass

import cv2
import supervision as sv
from supervision.dataset.utils import LazyLoadDict
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.helpers import split_data

from .detection_ontology import DetectionOntology

import shelve
import tempfile

@dataclass
class DetectionBaseModel(BaseModel):
    ontology: DetectionOntology

    @abstractmethod
    def predict(self, input: str) -> sv.Detections:
        pass

    def label(
        self, input_folder: str, extension: str = ".jpg", output_folder: str = None
    ) -> sv.DetectionDataset:
        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        images_map = LazyLoadDict()
        # Create a temporary file for the shelve
        temp_filename = tempfile.mktemp()
        detections_map = shelve.open(temp_filename)

        print(f"Storing temporary data in shelve file: {temp_filename}")

        files = glob.glob(input_folder + "/*" + extension)
        progress_bar = tqdm(files, desc="Labeling images")
        # iterate through images in input_folder
        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)
            image = cv2.imread(f_path)

            f_path_short = os.path.basename(f_path)
            images_map[f_path_short] = f_path
            detections = self.predict(f_path)
            detections_map[f_path_short] = detections

        detections_map.close() # Close the shelve file
        detections_map = shelve.open(temp_filename, flag='r')
        dataset = sv.DetectionDataset(
            self.ontology.classes(), images_map, detections_map
        )

        dataset.as_yolo(
            output_folder + "/images",
            output_folder + "/annotations",
            min_image_area_percentage=0.01,
            data_yaml_path=output_folder + "/data.yaml",
        )

        split_data(output_folder)

        print("Labeled dataset created - ready for distillation.")
        return dataset
