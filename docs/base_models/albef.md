[ALBEF](https://github.com/salesforce/LAVIS), developed by Salesforce, is a computer vision model that supports a range of tasks, including image-text pre-training, image-text retrieval, visual question anserting, and zero-shot classification. You can classify images using ALBEF with Autodistill.

## Installation

To use ALBEF with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-albef
```

## Quickstart

```python
from autodistill_albef import ALBEF

# define an ontology to map class names to our ALBEF prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = ALBEF(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```