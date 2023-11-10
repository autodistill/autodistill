import os
import urllib.request

import numpy as np
import torch
from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator

import supervision as sv

from dataclasses import dataclass
import cv2
from autodistill.detection import CaptionOntology, DetectionBaseModel

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


torch.use_deterministic_algorithms(False)

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class SegmentAnything(DetectionBaseModel):
    ontology: CaptionOntology
    box_threshold: float
    text_threshold: float

    def __init__(self, ontology: CaptionOntology, box_threshold=0.35, text_threshold=0.25):
        self.ontology = ontology
        self.predictor = load_SAM()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, input: str) -> sv.Detections:
        masks = self.predictor.generate(cv2.imread(input))

        results = sv.Detections.from_sam(masks)

        results.class_id = np.array([0] * len(results.mask[0]))

        return results

def load_SAM():
    # Check if segment-anything library is already installed

    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")
    SAM_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "samhq")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_DIR, "samhq.pth")

    url = "https://media.roboflow.com/samhq.pth"

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)

    # Download the file if it doesn't exist
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    SAM_ENCODER_VERSION = "vit_l/vit_b/vit_h/vit_tiny"

    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
        device=DEVICE
    )
    
    sam_predictor = SamAutomaticMaskGenerator(sam)

    return sam_predictor