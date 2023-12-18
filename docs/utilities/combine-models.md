You can combine detection, segmentation, and classification models to leverage the strengths of each model.

For example, consider a scenario where you want to build a logo detection model that identifies popular logos. You could use a detection model to identify logos (i.e. Grounding DINO), then a classification model to classify between the logos (i.e. Microsoft, Apple, etc.).

To combine models, you need to choose:

1. Either a detection or a segmentation model, and;
2. A classification model.

Let's walk through an example of using a combination of Grounding DINO and SAM (GroundedSAM), and CLIP for logo classification.

```python
from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import supervision as sv

from autodistill.core.custom_detection_model import CustomDetectionModel
import cv2

classes = ["McDonalds", "Burger King"]


SAMCLIP = CustomDetectionModel(
    detection_model=GroundedSAM(
        CaptionOntology({"logo": "logo"})
    ),
    classification_model=CLIP(
        CaptionOntology({k: k for k in classes})
    )
)

IMAGE = "logo.jpg"

results = SAMCLIP.predict(IMAGE)

image = cv2.imread(IMAGE)

annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{classes[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, _ in results
]

annotated_frame = annotator.annotate(
    scene=image.copy(), detections=results
)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame, labels=labels, detections=results
)

sv.plot_image(annotated_frame, size=(8, 8))
```

Here are the results:

![SAMCLIP Example](https://media.roboflow.com/autodistill/samclip.png)

## See Also

- [Automatically Label Product SKUs with Autodistill
](https://blog.roboflow.com/label-product-skus/): Uses a combination of Grounding DINO and CLIP to label product SKUs.

## Code Reference

:::autodistill.core.composed_detection_model.ComposedDetectionModel