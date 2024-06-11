<span class="cls-button">Classification</span>
<span class="bm-button">Base Model</span>
<span class="cm-button">Community Contribution</span>

## What is EvaCLIP?

_Note: This module was contributed by a third-party community member unaffiliated with Roboflow._ 

This repository contains the code supporting the EvaCLIP base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[EvaCLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP), is a computer vision model trained using pairs of images and text. It can be used for classification of images.

## Installation

To use EvaCLIP with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-evaclip
```

## Quickstart

```python
from autodistill_evaclip import EvaCLIP
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our EvaCLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = EvaCLIP(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)

results = base_model.predict("./context_images/test.jpg")

print(results)

base_model.label("./context_images", extension=".jpeg")
```
