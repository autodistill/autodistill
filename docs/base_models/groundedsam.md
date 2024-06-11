<span class="od-button">Object Detection</span>
<span class="bm-button">Base Model</span>

# What is Grounded SAM?

Grounded SAM uses the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) to identify objects in an image and assign labels to each image.

## Installation

To use the Grounded SAM base model, you will need to install the following dependency:

```bash
pip3 install autodistill-grounded-sam
```

## Quickstart

```python
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our Grounded SAM prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedSAM(ontology=CaptionOntology({"shipping container": "container"}))

# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
```
