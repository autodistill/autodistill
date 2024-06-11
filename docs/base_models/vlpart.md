<span class="od-button">Object Detection</span>
<span class="bm-button">Base Model</span>

# What is VLPart?

[VLPart](https://github.com/facebookresearch/VLPart), developed by Meta Research, is an object detection and segmentation model that works with an open vocabulary. `autodistill-vlpart` enables you to use VLPart to auto-label images for use in training a fine-tuned model. `autodistill-vlpart` supports the LVIS vocabulary.

## Installation

To use VLPart with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-vlpart
```

## Quickstart

```python
from autodistill_vlpart import VLPart
from autodistill.detection import CaptionOntology
from autodistill.utils import plot

# define an ontology to map class names to our VLPart prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = VLPart(
    ontology=CaptionOntology(
        {
            "person": "person"
        }
    )
)

predictions = base_model.predict("./image.png")

print(predictions)

plot(
    image=cv2.imread("./image.png"),
    classes=base_model.class_names,
    detections=predictions
)

# label the images in the context_images folder
base_model.label("./context_images", extension=".jpeg")
```

## License

This project is licensed under an [MIT license](https://github.com/autodistill/autodistill-vlpart/blob/main/LICENSE).