# Grounding DINO

[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) is a zero-shot object detection model developed by IDEA Research. You can distill knowledge from Grounding DINO into a smaller model using Autodistill.

## Installation

To use the Grounded dino base model, you will need to install the following dependency:

```bash
pip3 install autodistill-grounding-dino
```

## Quickstart

```python
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology


# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedSAM(ontology=CaptionOntology({"shipping container": "container"}))

# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
```