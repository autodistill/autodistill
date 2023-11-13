import numpy as np
import supervision as sv
from PIL import Image

from autodistill.detection.detection_base_model import DetectionBaseModel


class CustomDetectionModel(DetectionBaseModel):
    """
    Run inference with a detection model then run inference with a classification model on the detected regions.
    """

    def __init__(self, detection_model, classification_model, set_of_mark=None):
        self.detection_model = detection_model
        self.classification_model = classification_model
        self.set_of_mark = set_of_mark
        self.ontology = self.classification_model.ontology

    def predict(self, image: str) -> sv.Detections:
        """
        Run inference with a detection model then run inference with a classification model on the detected regions.

        :param detection_model: A detection model
        :param classification_model: A classification model
        :param image: Path to image

        :return: A list of detections
        """
        detections = []
        opened_image = Image.open(image)

        detections = self.detection_model.predict(image)

        if self.set_of_mark is not None:
            label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

            labels = [f"{num}" for num in range(len(detections.xyxy))]

            opened_image = np.array(opened_image)

            annotated_frame = label_annotator.annotate(
                scene=opened_image, labels=labels, detections=detections
            )

            opened_image = Image.fromarray(annotated_frame)

            opened_image.save("temp.jpeg")

            result = self.classification_model.predict("temp.jpeg")

            detections.class_id = result.class_id
            detections.confidence = result.confidence

            return detections

        for pred_idx, bbox in enumerate(detections.xyxy):
            # extract region from image
            region = opened_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            # save as tempfile
            region.save("temp.jpeg")

            result = self.classification_model.predict("temp.jpeg")

            if len(result.class_id) == 0:
                continue

            result = result.get_top_k(1)[0][0]

            detections.class_id[pred_idx] = result

        return detections
