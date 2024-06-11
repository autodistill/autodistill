<span class="cls-button">Object Detection</span>
<span class="tm-button">Target Model</span>

# What is DETR?

This repository contains the code supporting the DETR base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[DETR](https://huggingface.co/docs/transformers/model_doc/detr) is a transformer-based computer vision model you can use for object detection. Autodistill supports training a model using the Meta Research Resnet 50 checkpoint.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [DETR Autodistill documentation](https://autodistill.github.io/autodistill/target_models/detr/).

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

## License

This project is licensed under an [Apache 2.0 license](LICENSE). See the [Hugging Face model card for the DETR Resnet 50](https://huggingface.co/facebook/detr-resnet-50) model for more information on the model license.