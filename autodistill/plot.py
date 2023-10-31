import supervision as sv
import numpy as np

from typing import List

def plot(detections, image: np.ndarray, classes: List[str]):
    if detections.mask:
        annotator = sv.MaskAnnotator()
    else:
        annotator = sv.BoxAnnotator()

    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]

    annotated_frame = annotator.annotate(
        scene=image.copy(),
        detections=detections,
        labels=labels
    )