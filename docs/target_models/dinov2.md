# DINOv2

**CLIP is not fully supported by Autodistill. Check back later for updates.**

[DINOv2](https://github.com/facebookresearch/dinov2), developed by Meta Research, is a self-supervised training method for computer vision models. This library uses DINOv2 image embeddings with SVM to build a classification model.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [DINOv2 Autodistill documentation](https://autodistill.github.io/autodistill/base_models/dinov2/).

## Installation

To use DINOv2 with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-dinov2
```

## Quickstart

```python
from autodistill_dinov2 import DINOv2

target_model = DINOv2()

# train a model
# specify the directory where your annotations (in multiclass classification folder format)
# DINOv2 embeddings are saved in a file called "embeddings.json" the folder in which you are working
# with the structure {filename: embedding}
target_model.train("./context_images_labeled")

# run inference on the new model
pred = target_model.predict("./context_images_labeled/train/images/dog-7.jpg")
```