import supervision as sv
from PIL import Image


class CustomDetectionModel:
    """
    Run inference with a detection model then run inference with a classification model on the detected regions.
    """

    def __init__(self, detection_model, classification_model):
        self.detection_model = detection_model
        self.classification_model = classification_model

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
