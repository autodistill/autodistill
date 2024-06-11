<span class="od-button">Object Detection</span>
<span class="bm-button">Base Model</span>

## What is GPT-4o?

[GPT-4o](https://openai.com/index/hello-gpt-4o/), developed by OpenAI, is a multi-modal language model that works across the image, text, and audio domains. With GPT-4o, you can ask questions about images in natural language. The `autodistill-gpt4o` module enables you to classify images using GPT-4V.

This model uses the [gpt-4-o API](https://platform.openai.com/docs/guides/vision) announced by OpenAI on May 13th, 2024.

> [!NOTE]  
> Using this project will incur billing charges for API calls to the OpenAI GPT-4 Vision API.
> Refer to the [OpenAI pricing](https://openai.com/pricing) page for more information and to calculate your expected pricing. This package makes one API call per image you want to label.

## Installation

To use GPT-4o with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-gpt-4o
```

## Quickstart

```python
from autodistill_gpt_4o import GPT4o

# define an ontology to map class names to our GPT-4o prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GPT4o(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    ),
    api_key="OPENAI_API_KEY"
)
base_model.label("./context_images", extension=".jpeg")
```

## License

This project is licensed under an [MIT license](https://github.com/autodistill/autodistill-gpt-4o/blob/main/LICENSE).