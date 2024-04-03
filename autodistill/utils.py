from typing import List

import cv2
import numpy as np
import supervision as sv


def compare(models: list, images: List[str]):
    """
    Compare the predictions of multiple models on multiple images.

    Args:
        models: The models to compare
        images: The images to compare

    Returns:
        A grid of images with the predictions of each model on each image.
    """
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
    """
    Plot bounding boxes or segmentation masks on an image.

    Args:
        image: The image to plot on
        detections: The detections to plot
        classes: The classes to plot
        raw: Whether to return the raw image or plot it interactively

    Returns:
        The raw image (np.ndarray) if raw=True, otherwise None (image is plotted interactively
    """
    # TODO: When we have a classification annotator
    # in supervision, we can add it here
    if detections.mask is not None:
        annotator = sv.MaskAnnotator()
    else:
        annotator = sv.BoundingBoxAnnotator()

    label_annotator = sv.LabelAnnotator()

    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _, _ in detections
    ]

    annotated_frame = annotator.annotate(scene=image.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, labels=labels, detections=detections
    )

    if raw:
        return annotated_frame

    sv.plot_image(annotated_frame, size=(8, 8))
