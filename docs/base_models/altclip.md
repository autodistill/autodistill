<span class="cls-button">Classification</span>
<span class="bm-button">Base Model</span>

## What is AltCLIP?

[AltCLIP](https://arxiv.org/abs/2211.06679v2) is a multi-modal vision model. With AltCLIP, you can compare the similarity between text and images, or the similarlity between two images. AltCLIP was trained on multi-lingual text-image pairs, which means it can be used for zero-shot classification with text prompts in different languages. [Read the AltCLIP paper for more information](https://arxiv.org/pdf/2211.06679v2.pdf).

The Autodistill AltCLIP module enables you to use AltCLIP for zero-shot classification.

## Installation

To use AltCLIP with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-altclip
```

## Quickstart

```python
from autodistill_altclip import AltCLIP
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our AltCLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated results
# then, load the model
base_model = AltCLIP(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)

results = base_model.predict("construction.jpg")

print(results)
```

## License

The AltCLIP model is licensed under an [Apache 2.0 license](LICENSE). See the [model README](https://github.com/FlagAI-Open/FlagAI/blob/master/examples/AltCLIP/README.md) for more information.

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
