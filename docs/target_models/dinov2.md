<span class="cls-button">Classification</span>
<span class="bm-button">Base Model</span>

## What is DINOv2?

This repository contains the code supporting the DINOv2 base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[DINOv2](https://github.com/facebookresearch/dinov2), developed by Meta Research, is a self-supervised training method for computer vision models. This library uses DINOv2 image embeddings with SVM to build a classification model.

## Installation

To use DINOv2 with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-dinov2
```

## Quickstart

```python
from autodistill_dinov2 import DINOv2

target_model = DINOv2(None)

# train a model
# specify the directory where your annotations (in multiclass classification folder format)
# DINOv2 embeddings are saved in a file called "embeddings.json" the folder in which you are working
# with the structure {filename: embedding}
target_model.train("./context_images_labeled")

# get class list
# print(target_model.ontology.classes())

# run inference on the new model
pred = target_model.predict("./context_images_labeled/train/images/dog-7.jpg")

print(pred)
```
