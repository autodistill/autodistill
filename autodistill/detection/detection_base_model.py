import glob
import os
from abc import abstractmethod
from dataclasses import dataclass

import cv2
import supervision as sv
from tqdm import tqdm

from autodistill.core import BaseModel
from autodistill.detection import DetectionOntology
from autodistill.helpers import split_data

import math
import shutil

@dataclass
class DetectionBaseModel(BaseModel):
    ontology: DetectionOntology

    @abstractmethod
    def predict(self, input: str) -> sv.Detections:
        pass

    def label(
        self, input_folder: str, extension: str = ".jpg", output_folder: str = None, chunks: int = 1
    ) -> sv.DetectionDataset:
        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)
        
        if not os.path.exists(os.path.join(output_folder, "data.yaml")): # Do the labeling because there are no labels
            images_map = {}
            detections_map = {}

            files = glob.glob(input_folder + "/*" + extension)
            
            ######### CHUNK PROCESSING NEW
            # Compute the size of each chunk
            chunk_size = math.ceil(len(files) / chunks)
            
            all_chunk_folders = []
            
            # Break files into smaller chunks for processing
            for i in range(chunks):
                chunk_output_folder = f"{output_folder}_chunk_{i+1}"
                all_chunk_folders.append(chunk_output_folder)
                os.makedirs(chunk_output_folder, exist_ok=True)
                
                if os.path.exists(os.path.join(chunk_output_folder, "data.yaml")):
                    continue # Skip this chunk because it is already labeled

                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(files))

                current_files_chunk = files[start_idx:end_idx]
                progress_bar = tqdm(current_files_chunk, desc=f"Labeling Chunk {i+1}/{chunks}")

                images_map = {}  
                detections_map = {} 

                # Process each chunk
                for f_path in progress_bar:
                    progress_bar.set_description(desc=f"Labeling Chunk {i+1}/{chunks}: {f_path}", refresh=True)
                    image = cv2.imread(f_path)

                    f_path_short = os.path.basename(f_path)
                    images_map[f_path_short] = image.copy()
                    detections = self.predict(f_path)
                    detections_map[f_path_short] = detections

                dataset = sv.DetectionDataset(
                    self.ontology.classes(), images_map, detections_map
                )

                dataset.as_yolo(
                    chunk_output_folder + "/images",
                    chunk_output_folder + "/annotations",
                    min_image_area_percentage=0.01,
                    data_yaml_path=chunk_output_folder + "/data.yaml",
                )

                del images_map
                del detections_map
                
            # After processing all chunks, aggregate data into main output folder
            for chunk_folder in tqdm(all_chunk_folders, desc="Building final dataset"):
                for subdir, _, files in os.walk(chunk_folder):
                    for file_name in files:
                        source = os.path.join(subdir, file_name)
                        dest = os.path.join(output_folder, os.path.relpath(subdir, chunk_folder), file_name)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        shutil.copy2(source, dest)

        split_data(output_folder)

        print("Labeled dataset created - ready for distillation.")
        return dataset
