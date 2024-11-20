import enum
import glob
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import roboflow
import supervision as sv
from PIL import Image
from supervision.utils.file import save_text_file
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.helpers import load_image, split_data

from .detection_ontology import DetectionOntology


class NmsSetting(str, enum.Enum):
    NONE = "no_nms"
    CLASS_SPECIFIC = "class_specific"
    CLASS_AGNOSTIC = "class_agnostic"


@dataclass
class DetectionBaseModel(BaseModel):
    ontology: DetectionOntology

    @abstractmethod
    def predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
        pass

    def sahi_predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
        slicer = sv.InferenceSlicer(callback=self.predict)

        return slicer(load_image(input, return_format="cv2"))

    def _record_confidence_in_files(
        self,
        annotations_directory_path: str,
        image_names: List[str],
        annotations: Dict[str, sv.Detections],
    ) -> None:
        Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)
        for image_name in image_names:
            detections = annotations[image_name]
            yolo_annotations_name, _ = os.path.splitext(image_name)
            confidence_path = os.path.join(
                annotations_directory_path,
                "confidence-" + yolo_annotations_name + ".txt",
            )
            if detections.confidence is None:
                raise ValueError("Expected detections to have confidence values.")
            confidence_list = [str(x) for x in detections.confidence.tolist()]
            save_text_file(lines=confidence_list, file_path=confidence_path)
            print("Saved confidence file: " + confidence_path)

    def label(
        self,
        input_folder: str,
        extension: str = ".jpg",
        output_folder: str | None = None,
        human_in_the_loop: bool = False,
        roboflow_project: str | None = None,
        roboflow_tags: list[str] = ["autodistill"],
        sahi: bool = False,
        record_confidence: bool = False,
        nms_settings: NmsSetting = NmsSetting.NONE,
    ) -> sv.DetectionDataset:
        """
        Label a dataset with the model.
        """
        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        image_paths = glob.glob(input_folder + "/*" + extension)
        detections_map = {}

        if sahi:
            slicer = sv.InferenceSlicer(callback=self.predict)

        progress_bar = tqdm(image_paths, desc="Labeling images")
        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)

            image = cv2.imread(f_path)
            if sahi:
                detections = slicer(image)
            else:
                detections = self.predict(image)

            if nms_settings == NmsSetting.CLASS_SPECIFIC:
                detections = detections.with_nms()
            if nms_settings == NmsSetting.CLASS_AGNOSTIC:
                detections = detections.with_nms(class_agnostic=True)

            detections_map[f_path] = detections

        dataset = sv.DetectionDataset(
            self.ontology.classes(), image_paths, detections_map
        )

        dataset.as_yolo(
            output_folder + "/images",
            output_folder + "/annotations",
            min_image_area_percentage=0.01,
            data_yaml_path=output_folder + "/data.yaml",
        )

        if record_confidence:
            image_names = [os.path.basename(f_path) for f_path in image_paths]
            self._record_confidence_in_files(
                output_folder + "/annotations", image_names, detections_map
            )
        split_data(output_folder, record_confidence=record_confidence)

        if human_in_the_loop:
            roboflow.login()

            rf = roboflow.Roboflow()

            workspace = rf.workspace()

            workspace.upload_dataset(output_folder, project_name=roboflow_project)

        print("Labeled dataset created - ready for distillation.")
        return dataset
