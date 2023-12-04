You can compare two or more models on multiple images using the `compare` function.

This function is ideal if you want to evaluate how different models perform on a single image or multiple images.

The following example shows how to compare OWLv2 and Grounding DINO on a single image:

```python
from autodistill_grounding_dino import GroundingDINO
from autodistill_owlv2 import OWLv2

from autodistill.detection import CaptionOntology
from autodistill.utils import compare

ontology = CaptionOntology(
    {
        "solar panel": "solar panel",
    }
)

models = [
    GroundingDINO(ontology=ontology),
    OWLv2(ontology=ontology),
]

images = [
    "./solar.jpg"
]

compare(
    models=models,
    images=images
)
```

Here are the results:

![Compare Example](https://media.roboflow.com/autodistill/compare.png)

Above, we can see predictions from Grounding DINO and OWLv2.

## Code Reference

:::autodistill.utils.compare