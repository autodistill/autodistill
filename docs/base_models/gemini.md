<span class="cls-button">Classification</span>
<span class="bm-button">Base Model</span>

## What is Gemini?

This repository contains the code supporting the Gemini base model for use with [Autodistill](https://github.com/autodistill/autodistill-gemini).

[Gemini](https://blog.google/technology/ai/google-gemini-ai), family of models, representing the next generation of highly
compute-efficient multimodal models capable of recalling and reasoning over fine-grained information
from millions of tokens of context, including multiple long documents and hours of video and audio.s.

## Installation

To use Gemini with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-gemini
```

## Quickstart

```python
from autodistill_gemini import Gemini

# define an ontology to map class names to our Gemini prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = Gemini(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    ),
    gcp_region="us-central1",
    gcp_project="project-name",
    model="gemini-1.5-flash"
)

# run inference on an image
result = base_model.predict("image.jpg")

print(result)

# label a folder of images
base_model.label("./context_images", extension=".jpeg")
```
