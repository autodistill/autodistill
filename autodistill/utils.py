from typing import List

import cv2
import numpy as np
import supervision as sv


def compare(models: list, images: List[str]):
    image_results = []
    model_results = []

    for model in models:
        # get model class name
        model_name = model.__class__.__name__

        for image in images:
            results = model.predict(image)

            image_data = cv2.imread(image)

            image_result = plot(
                image_data, results, classes=model.ontology.prompts(), raw=True
            )

            image_results.append(image_result)

            model_results.append(model_name)

    sv.plot_images_grid(
        image_results,
        grid_size=(len(models), len(images)),
        titles=model_results,
        size=(16, 16),
    )


def plot(image: np.ndarray, detections, classes: List[str], raw=False):
    # TODO: When we have a classification annotator
    # in supervision, we can add it here
    if detections.mask:
        annotator = sv.MaskAnnotator()
    else:
        annotator = sv.BoxAnnotator()

    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _ in detections
    ]

    annotated_frame = annotator.annotate(
        scene=image.copy(), detections=detections, labels=labels
    )

    if raw:
        return annotated_frame

    sv.plot_image(annotated_frame, size=(8, 8))
