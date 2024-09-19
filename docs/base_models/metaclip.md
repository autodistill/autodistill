<span class="cls-button">Object Detection</span>
<span class="bm-button">Base Model</span>

## What is MetaClip?

[MetaCLIP](https://github.com/facebookresearch/MetaCLIP), developed by Meta AI Research, is a computer vision model trained using pairs of images and text. The model was described in the [Demystifying CLIP Data](https://arxiv.org/abs/2309.16671) paper. You can use MetaCLIP with autodistill for image classification.

## Installation

To use MetaCLIP with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-metaclip
```

## Quickstart

### get predictions

```python
from autodistill_metaclip import MetaCLIP

# define an ontology to map class names to our MetaCLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = MetaCLIP(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)

results = base_model.predict("./image.png")
print(results)
```

### calculate and compare embeddings

```python
from autodistill_metaclip import MetaCLIP

base_model = MetaCLIP(None)

text = base_model.embed_text("coffee")
image = base_model.embed_image("coffeeshop.jpg")

print(base_model.compare(text, image))
```
