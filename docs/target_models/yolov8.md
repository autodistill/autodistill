<span class="cls-button">Object Detection</span>
<span class="sm-button">Segmentation</span>
<span class="tm-button">Target Model</span>

# What is YOLOv8?

Ultralytics YOLOv8 is a Convolutional Neural Network (CNN) that supports realtime object detection, instance segmentation, and other tasks. It can be deployed to a variety of edge devices.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [YOLOv8 Autodistill documentation](https://autodistill.github.io/autodistill/target_models/yolov8/).

## Installation

To use the YOLOv8 Target Model, simply install it along with a Base Model supporting the `detection` task:

```bash
pip3 install autodistill-grounded-sam autodistill-yolov8
```

You can find a full list of `detection` Base Models on [the main autodistill repo](https://github.com/autodistill/autodistill).

## Quickstart (Train a YOLOv8 Model)

```python
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
base_model = GroundedSAM(ontology=CaptionOntology({"shipping container": "container"}))

# label all images in a folder called `context_images`
base_model.label(
  input_folder="./images",
  output_folder="./dataset"
)

target_model = YOLOv8("yolov8n.pt")
target_model.train("./dataset/data.yaml", epochs=200)

# run inference on the new model
pred = target_model.predict("./dataset/valid/your-image.jpg", confidence=0.5)
print(pred)

# optional: upload your model to Roboflow for deployment
from roboflow import Roboflow

rf = Roboflow(api_key="API_KEY")
project = rf.workspace().project("PROJECT_ID")
project.version(DATASET_VERSION).deploy(model_type="yolov8", model_path=f"./runs/detect/train/")
```

## Quickstart (Use a YOLOv8 Model to Label Data)

```python
from autodistill_yolov8 import YOLOv8Base
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our YOLOv8 classes
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model

# replace weights_path with the path to your YOLOv8 weights file
base_model = YOLOv8Base(ontology=CaptionOntology({"car": "car"}), weights_path="yolov5s.pt")

# run inference on a single image
results = base_model.predict("mercedes.jpeg")

base_model.label(
  input_folder="./images",
  output_folder="./dataset"
)
```

## Choosing a Task

YOLOv8 supports training both object detection and instance segmentation tasks at various sizes (larger models are slower but can be more accurate). This selection is done in the constructor.

For example:
```python
# initializes a nano-sized instance segmentation model
target_model = YOLOv8("yolov8n-seg.pt")
```

Available object detection initialization options are:

* `yolov8n.pt` - nano (3.2M parameters)
* `yolov8s.pt` - small (11.2M parameters)
* `yolov8m.pt` - medium (25.9M parameters)
* `yolov8l.pt` - large (43.7M parameters)
* `yolov8x.pt` - extra-large (68.2M parameters)

Available instance segmentation initialization options are:

* `yolov8n-seg.pt` - nano (3.4M parameters)
* `yolov8s-seg.pt` - small (11.8M parameters)
* `yolov8m-seg.pt` - medium (27.3M parameters)
* `yolov8l-seg.pt` - large (46.0M parameters)
* `yolov8x-seg.pt` - extra-large (71.8M parameters)

## License

The code in this repository is licensed under an [AGPL 3.0 license](https://github.com/autodistill/autodistill-yolov8/edit/main/LICENSE).