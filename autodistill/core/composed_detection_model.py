import numpy as np
import supervision as sv
from PIL import Image

from autodistill.detection.detection_base_model import DetectionBaseModel

DEFAULT_LABEL_ANNOTATOR = sv.LabelAnnotator(text_position=sv.Position.CENTER)
SET_OF_MARKS_SUPPORTED_MODELS = ["GPT4V"]


class ComposedDetectionModel(DetectionBaseModel):
    """
    Run inference with a detection model then run inference with a classification model on the detected regions.
    """

    def __init__(
        self,
        detection_model,
        classification_model,
        set_of_marks=None,
        set_of_marks_annotator=DEFAULT_LABEL_ANNOTATOR,
    ):
        self.detection_model = detection_model
        self.classification_model = classification_model
        self.set_of_marks = set_of_marks
        self.set_of_marks_annotator = set_of_marks_annotator
        self.ontology = self.classification_model.ontology

    def predict(self, image: str) -> sv.Detections:
        """
        Run inference with a detection model then run inference with a classification model on the detected regions.

        Args:
            image: The image to run inference on
            annotator: The annotator to use to annotate the image

        Returns:
            detections (sv.Detections)
        """
        detections = []
        opened_image = Image.open(image)

        detections = self.detection_model.predict(image)

        if self.set_of_marks is not None:
            labels = [f"{num}" for num in range(len(detections.xyxy))]

            opened_image = np.array(opened_image)

            annotated_frame = self.set_of_marks_annotator.annotate(
                scene=opened_image, labels=labels, detections=detections
            )

            opened_image = Image.fromarray(annotated_frame)

            opened_image.save("temp.jpeg")

            if not hasattr(self.classification_model, "set_of_marks"):
                raise Exception(
                    f"The set classification model does not have a set_of_marks method. Supported models: {SET_OF_MARKS_SUPPORTED_MODELS}"
                )

            result = self.classification_model.set_of_marks(
                input=image, masked_input="temp.jpeg", classes=labels, masks=detections
            )

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
