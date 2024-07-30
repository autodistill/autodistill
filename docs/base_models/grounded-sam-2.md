<span class="is-button">Image Segmentation</span>
<span class="bm-button">Base Model</span>

# Grounded SAM 2 Base Model

This repository contains the code implementing Grounded SAM 2 using Florence-2 as a grounding model and Segment Anything 2 as a segmentation model for use with [`autodistill`](https://github.com/autodistill/autodistill).

Florence-2 is a zero-shot multimodal model. You can use Florence-2 for open vocabulary object detection. This project uses the object detection capabilities in Florence-2 to ground the SAM 2 model.

## Installation

To use the Grounded SAM 2 base model, you will need to install the following dependency:

```bash
pip3 install autodistill-grounded-sam-2
```

## Quickstart

```python
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our Grounded SAM 2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedSAM2(
    ontology=CaptionOntology(
        {
            "person": "person",
            "shipping container": "shipping container",
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
# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
```

## License

The code in this repository is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
