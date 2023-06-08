# YOLO-NAS

[YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md) is an object detection model developed by [Deci AI](https://deci.ai/).

## Installation

To use the YOLO-NAS target model, you will need to install the following dependency:

```bash
pip3 install autodistill-yolo-nas
```

## Quickstart

```python
from autodistill_yolo_nas import YOLONAS

target_model = YOLONAS()

# train a model
# specify the directory where your annotations (in YOLO format) are stored
target_model.train("./context_images_labeled", epochs=20)

# run inference on the new model
pred = target_model.predict("./context_images_labeled/train/images/dog-7.jpg", confidence=0.01)
```