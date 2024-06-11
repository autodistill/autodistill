<span class="sm-button">Segmentation</span>
<span class="bm-button">Base Model</span>

# What is EvaCLIP?

[EdgeSAM](https://github.com/chongzhou96/EdgeSAM), introduced in the "EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM" paper, is a faster version of the Segment Anything model.

Grounded EdgeSAM combines [Grounding DINO](https://blog.roboflow.com/grounding-dino-zero-shot-object-detection/) and EdgeSAM, allowing you to identify objects and generate segmentation masks for them.

## Installation

To use Grounded EdgeSAM with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-grounded-edgesam
```

## Quickstart

```python
from autodistill_clip import CLIP

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
from autodistill_grounded_edgesam import GroundedEdgeSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our GroundedSAM prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedEdgeSAM(
    ontology=CaptionOntology(
        {
            "person": "person",
            "forklift": "forklift",
        }
    )
)

# run inference on a single image
results = base_model.predict("logistics.jpeg")

plot(
    image=cv2.imread("logistics.jpeg"),
    classes=base_model.ontology.classes(),
    detections=results
)

# label a folder of images
base_model.label("./context_images", extension=".jpeg")
```

## License

This repository is released under an [S-Lab License 1.0](https://github.com/autodistill/autodistill-grounded-edgesam/blob/main/LICENSE) license.
