<span class="od-button">Object Detection</span>
<span class="bm-button">Base Model</span>

_This model has a newer version: [OWLv2](/base_models/owlv2/). We recommend using OWLv2 for better performance._

# What is OWL-ViT?

[OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) is a transformer-based object detection model developed by Google Research.

## Installation

To use OWL-ViT with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-owl-vit
```

## Quickstart

```python
from autodistill_owl_vit import OWLViT
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our OWLViT prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = OWLViT(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpg")
```
