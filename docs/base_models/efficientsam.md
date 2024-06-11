<span class="sm-button">Segmentation</span>
<span class="bm-button">Base Model</span>

## What is EfficientSAM?

This repository contains the code supporting the EfficientSAM base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[EfficientSAM](https://github.com/yformer/EfficientSAM) is an image segmentation model that was introduced in the paper "[EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything](https://yformer.github.io/efficient-sam/)". You can use EfficientSAM with autodistill for image segmentation.

## Installation

To use EfficientSAM with Autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-efficientsam
```

## Quickstart

This model returns segmentation masks for all objects in an image.

If you want segmentation masks only for specific objects matching a text prompt, we recommend combining EfficientSAM with a zero-shot detection model like GroundingDINO.

Read our ComposedDetectionModel documentation for more information about how to combine models like EfficientSAM and GroundingDINO.

```python
from autodistill_efficientsam import EfficientSAM

# define an ontology to map class names to our EfficientSAM prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = EfficientSAM(None)

masks = base_model.predict("./image.png")
```

## License

This project is licensed under an [Apache 2.0 license](https://github.com/autodistill/autodistill-efficientsam/blob/main/LICENSE).