<span class="cls-button">Classification</span>
<span class="tm-button">Target Model</span>

## What is ViT?

[ViT](https://huggingface.co/google/vit-base-patch16-224-in21k) is a classification model pre-trained on ImageNet-21k, developed by Google. You can train ViT classification models using Autodistill.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [ViT Autodistill documentation](https://autodistill.github.io/autodistill/target_models/vit/).

## Installation

To use the ViT target model, you will need to install the following dependency:

```bash
pip3 install autodistill-vit
```

## Quickstart

```python
from autodistill_vit import ViT

target_model = ViT()

# train a model from a classification folder structure
target_model.train("./context_images_labeled/", epochs=200)

# run inference on the new model
pred = target_model.predict("./context_images_labeled/train/images/dog-7.jpg", conf=0.01)
```
