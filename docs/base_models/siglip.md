<span class="cls-button">Classification</span>
<span class="bm-button">Base Model</span>

## What is ALBEF?

SigLIP is an image classification and embedding model architecture first introduced in the paper "[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)".

You can use SigLIP to classify images with Autodistill.

## Installation

To use SigLIP with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-siglip
```

## Quickstart

```python
from autodistill_siglip import SigLIP
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our SigLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
labels = ["person", "a forklift"]
base_model = SigLIP(
    ontology=CaptionOntology({item: item for item in labels})
)

results = base_model.predict("image.jpeg", confidence=0.1)

top_1 = results.get_top_k(1)

# show top label
print(labels[top_1[0][0]])

# label folder of images
base_model.label("./context_images", extension=".jpeg")
```


## License

The SigLIP model is licensed under an [Apache 2.0 license](https://huggingface.co/google/siglip-base-patch16-224).