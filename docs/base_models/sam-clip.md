<span class="sm-button">Segmentation</span>
<span class="bm-button">Base Model</span>

# What is SAM-CLIP?

SAM-CLIP uses the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) to identify objects in an image and assign labels to each image. Then, CLIP is used to find masks that are related to the given prompt.

## Installation

To use the SAM-CLIP base model, you will need to install the following dependency:

```bash
pip3 install autodistill-sam-clip
```

## Quickstart

```python
from autodistill_sam_clip import SAMCLIP
from autodistill.detection import CaptionOntology


# define an ontology to map class names to our CLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = SAMCLIP(ontology=CaptionOntology({"shipping container": "container"}))

# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
```