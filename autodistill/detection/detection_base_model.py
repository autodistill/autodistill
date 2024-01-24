import datetime
import glob
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import roboflow
import supervision as sv
from supervision.utils.file import save_text_file
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.helpers import load_image, split_data

from .detection_ontology import DetectionOntology


@dataclass
class DetectionBaseModel(BaseModel):
    ontology: DetectionOntology

    @abstractmethod
    def predict(self, input: str) -> sv.Detections:
        pass

    def sahi_predict(self, input: str) -> sv.Detections:
        slicer = sv.InferenceSlicer(callback=self.predict)

        return slicer(load_image(input, return_format="cv2"))

    def _record_confidence_in_files(
        self,
        annotations_directory_path: str,
        images: Dict[str, np.ndarray],
        annotations: Dict[str, sv.Detections],
    ) -> None:
        Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
        for image_name, _ in images.items():
            detections = annotations[image_name]
            yolo_annotations_name, _ = os.path.splitext(image_name)
            confidence_path = os.path.join(
                annotations_directory_path,
                "confidence-" + yolo_annotations_name + ".txt",
            )
            confidence_list = [str(x) for x in detections.confidence.tolist()]
            save_text_file(lines=confidence_list, file_path=confidence_path)
            print("Saved confidence file: " + confidence_path)

    def label(
        self,
        input_folder: str,
        extension: str = ".jpg",
        extensions: list = None,
        recursive: bool = False,
        output_folder: str = None,
        human_in_the_loop: bool = False,
        roboflow_project: str = None,
        roboflow_tags: str = ["autodistill"],
        sahi: bool = False,
        record_confidence: bool = False,
        with_nms: bool = False,
    ) -> sv.DetectionDataset:
        """
        Label a dataset with the model.
        """
        
      # Use 'extensions', fall back to 'extension'
        if extensions is not None:
            if extension != ".jpg":
                raise ValueError("`extension` and `extensions` are mutually exclusive.")
        else:
            extensions = [extension]

        files = []
        # Build file search pattern
        pattern = "/**/*{}" if recursive else "/*{}"
        for ext in extensions:
            search_pattern = os.path.join(input_folder, pattern.format(ext))
            found_files = glob.glob(search_pattern, recursive=recursive)
            files.extend(found_files)

        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        images_map = {}
        detections_map = {}

        if sahi:
            slicer = sv.InferenceSlicer(callback=self.predict)

        progress_bar = tqdm(files, desc="Labeling images")
        # iterate through images in input_folder
        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)
            image = cv2.imread(f_path)

            f_path_short = os.path.basename(f_path)
            images_map[f_path_short] = image.copy()

            if sahi:
                detections = slicer(f_path)
            else:
                detections = self.predict(f_path)

            if with_nms:
                detections = detections.with_nms()

            detections_map[f_path_short] = detections

        dataset = sv.DetectionDataset(
            self.ontology.classes(), images_map, detections_map
        )

        dataset.as_yolo(
            output_folder + "/images",
            output_folder + "/annotations",
            min_image_area_percentage=0.01,
            data_yaml_path=output_folder + "/data.yaml",
        )

        if record_confidence is True:
            self._record_confidence_in_files(
                output_folder + "/annotations", images_map, detections_map
            )
        split_data(output_folder, record_confidence=record_confidence)

        if human_in_the_loop:
            roboflow.login()

            rf = roboflow.Roboflow()

            workspace = rf.workspace()

            workspace.upload_dataset(output_folder, project_name=roboflow_project)

        print("Labeled dataset created - ready for distillation.")
        return dataset
