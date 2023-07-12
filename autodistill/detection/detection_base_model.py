import glob
import os
from abc import abstractmethod
from dataclasses import dataclass

import cv2
import supervision as sv
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.detection.detection_ontology import DetectionOntology
from autodistill.helpers import split_data
from roboflow import Roboflow
from tqdm import tqdm



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
        roboflow_api_key: str = None,
        roboflow_workspace: str = None,
        roboflow_project: str = None,
    ) -> sv.DetectionDataset:
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

        if (
            human_in_the_loop
            and roboflow_api_key is not None
            and roboflow_workspace is not None
            and roboflow_project is not None
        ):
            rf = Roboflow(api_key=roboflow_api_key)
            workspace = rf.workspace(roboflow_workspace)
            
            workspace.upload_dataset(output_folder, project_name=roboflow_project)

        print("Labeled dataset created - ready for distillation.")
        return dataset
