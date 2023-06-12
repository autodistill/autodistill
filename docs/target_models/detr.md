# DETR

[DETR](https://huggingface.co/docs/transformers/model_doc/detr) is a transformer-based computer vision model you can use for object detection. Autodistill supports training a model using the Meta Research Resnet 50 checkpoint.

## Installation

To use DETR with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-detr
```

## Quickstart

```python
from autodistill_detr import DETR

# load the model
target_model = DETR()

# train for 10 epochs
target_model.train("./roads", epochs=10)

# run inference on an image
target_model.predict("./roads/valid/-3-_jpg.rf.bee113a09b22282980c289842aedfc4a.jpg")
```