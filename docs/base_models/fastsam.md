<span class="od-button">Object Detection</span>
<span class="bm-button">Base Model</span>

# What is FastSAM?

[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) is a segmentation model trained on 2% of the SA-1B dataset used to train the [Segment Anything Model](https://github.com/facebookresearch/segment-anything).

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [FastSAM Autodistill documentation](https://autodistill.github.io/autodistill/base_models/fastsam/).

## Installation

To use FastSAM with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-fastsam
```

## Quickstart

```python
from autodistill_fastsam import FastSAM

# define an ontology to map class names to our FastSAM prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = FastSAM(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```