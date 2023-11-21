import datetime
import glob
import os
from abc import abstractmethod
from dataclasses import dataclass

import cv2
import roboflow
import supervision as sv
from tqdm import tqdm
import json
import time

from autodistill.core import BaseModel
from autodistill.helpers import split_data

from .detection_ontology import DetectionOntology


@dataclass
class DetectionBaseModel(BaseModel):
    ontology: DetectionOntology

    @abstractmethod
    def predict(self, input: str) -> sv.Detections:
        pass

    def label(
        self,
        input_folder: str,
        extension: str = ".jpg",
        output_folder: str = None,
        human_in_the_loop: bool = False,
        roboflow_project: str = None,
        roboflow_tags: str = ["autodistill"],
    ) -> sv.DetectionDataset:
        # call super.label to create output_folder
        output_folder, config = super().label(
            input_folder, extension, output_folder
        )

        images_map = {}
        detections_map = {}

        # if output_folder/autodistill.json exists
        if os.path.exists(output_folder + "/data.yaml"):
            dataset = sv.DetectionDataset.from_yolo(
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

        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)
            image = cv2.imread(f_path)

            f_path_short = os.path.basename(f_path)
            images_map[f_path_short] = image.copy()

            annotation_path = os.path.join(output_folder, "annotations/", ".".join(f_path_short.split(".")[:-1]) + ".txt")

            if not os.path.exists(annotation_path):
                detections = self.predict(f_path)
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

        split_data(output_folder)

        if human_in_the_loop:
            roboflow.login()

            rf = roboflow.Roboflow()

            workspace = rf.workspace()

            workspace.upload_dataset(output_folder, project_name=roboflow_project)

        config["end_time"] = time.time()
        
        with open(os.path.join(output_folder, "config.json"), "w+") as f:
            json.dump(config, f)

        print("Labeled dataset created - ready for distillation.")

        return dataset, output_folder
