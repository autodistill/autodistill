# OWL-ViT

**OWL-ViT is not fully supported by Autodistill. Check back later for updates.**

[OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) is a transformer-based object detection model developed by Google Research.

## Installation

To use OWL-ViT with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-owl-vit
```

## Quickstart

```python
from autodistill_owl_vit import OWLViT

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
