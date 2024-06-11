<span class="cls-button">Classification</span>
<span class="bm-button">Base Model</span>

## What is RemoteCLIP?

[RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) is a vision-language CLIP model trained on remote sensing data. According to the RemoteCLIP README:

> RemoteCLIP outperforms previous SoTA by 9.14% mean recall on the RSICD dataset and by 8.92% on RSICD dataset. For zero-shot classification, our RemoteCLIP outperforms the CLIP baseline by up to 6.39% average accuracy on 12 downstream datasets.

## Installation

To use RemoteCLIP with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-remote-clip
```

## Quickstart

```python
from autodistill_remote_clip import RemoteCLIP
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our RemoteCLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = RemoteCLIP(
    ontology=CaptionOntology(
        {
            "airport runway": "runway",
            "countryside": "countryside",
        }
    )
)

predictions = base_model.predict("runway.jpg")

print(predictions)
```

## License

This project is covered under an [Apache 2.0 license](https://github.com/ChenDelong1999/RemoteCLIP/blob/main/LICENSE).