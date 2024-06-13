<span class="od-button">Object Detection</span>
<span class="bm-button">Base Model</span>

## What is PaliGemma?

[PaLiGemma](https://blog.roboflow.com/paligemma-multimodal-vision/), developed by Google, is a computer vision model trained using pairs of images and text.

You can label data with PaliGemma models for use in training smaller, fine-tuned models with Autodisitll.

You can also fine-tune PaliGemma models with Autodistill.

## Installation

To use PaLiGemma with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-paligemma
```

## Quickstart

### Auto-label with an existing model

```python
from autodistill_paligemma import PaliGemma

# define an ontology to map class names to our PaliGemma prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = PaliGemma(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)

# label a single image
result = PaliGemma.predict("test.jpeg")
print(result)

# label a folder of images
base_model.label("./context_images", extension=".jpeg")
```

### Model fine-tuning

You can fine-tune PaliGemma models with LoRA for deployment with [Roboflow Inference](https://inference.roboflow.com).

To train a model, use this code:

```python
from autodistill_paligemma import PaLiGemmaTrainer

target_model = PaLiGemmaTrainer()
target_model.train("./data/")

result = target_model.predict("test.jpeg")
print(result)
```

## License

The model weights for PaLiGemma are licensed under a custom Google license. To learn more, refer to the [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms).
