# CLIP

[CLIP](https://github.com/openai/CLIP), developed by OpenAI, is a computer vision model trained using pairs of images and text. You can use CLIP with autodistill for image classification.

This project is licensed under an [MIT license](https://github.com/autodistill/autodistill-clip).

## Installation

To use CLIP with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-clip
```

## Quickstart

```python
from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = CLIP(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```