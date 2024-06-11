<span class="od-button">Object Detection</span>
<span class="bm-button">Base Model</span>

# What is YOLO-World?

[YOLO-World](https://github.com/AILab-CVC/YOLO-World) is a YOLO-based open vocabulary model for open vocabulary detection.

YOLO-World was developed by Tencent's AI Lab.

## Installation

To use YOLO-World with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-yolo-world
```
## Quickstart

```python
from autodistill_yolo_world import YOLOWorldModel
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our GroundedSAM prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = YOLOWorldModel(
    ontology=CaptionOntology(
        {
            "person": "person",
            "car": "car",
        }
    ),
    model_type = "yolov8s-world.pt"
)

# run inference on a single image
results = base_model.predict("assets/test.jpg")

plot(
    image=cv2.imread("assets/test.jpg"),
    classes=base_model.ontology.classes(),
    detections=results
)
# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
```
