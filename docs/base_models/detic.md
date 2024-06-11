<span class="od-button">Object Detection</span>
<span class="bm-button">Base Model</span>

# What is DETIC?

[DETIC](https://github.com/facebookresearch/Detic) is a transformer-based object detection and segmentation model developed by Meta Research.

## Installation

To use DETIC with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-detic
```

## Quickstart

```python
from autodistill_detic import DETIC
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our DETIC prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = DETIC(
    ontology=CaptionOntology(
        {
            "person": "person",
        }
    )
)
base_model.label("./context_images", extension=".jpg")
```