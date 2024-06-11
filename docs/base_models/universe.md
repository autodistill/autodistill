<span class="od-button">Object Detection</span>
<span class="sm-button">Segmentation</span>
<span class="bm-button">Base Model</span>

## What is Roboflow Universe?

[Roboflow Universe](https://universe.roboflow.com) is a community where people share computer vision models and datasets. Over 50,000 models and 250,000 datasets have been shared on Universe, with new models available every day. You can use Autodistill to run object detection and segmentation models hosted on Roboflow Universe.

> [!NOTE]
> Using this project will use Roboflow API calls. You will need a free Roboflow account to use this project. [Sign up for a free Roboflow account](https://app.roboflow.com) to get started. [Learn more about pricing](https://roboflow.com/pricing).

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [Roboflow Universe Autodistill documentation](https://autodistill.github.io/autodistill/base_models/roboflow_universe/).

## Installation

To use models hosted on Roboflow Universe with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-roboflow-universe
```

## Quickstart

> [!NOTE]
> Autodistill uses ontology to map model predictions to the expected class labels. For other Autodistill models, the term 'caption' is used when the model accepts prompting or a description for a prediction. When using Roboflow Universe as an Autodistill base model, the 'caption' will be the class name/label that the Universe model will return. 

```python
from autodistill_roboflow_universe import RoboflowUniverseModel
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our Roboflow model prompt:
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved in the generated annotations

model_configs = [
    ("PROJECT_ID", VERSION_NUMBER)
]

base_model = RoboflowUniverseModel(
    ontology=CaptionOntology(
        {
            "person": "person",
            "forklift": "vehicle"
        }
),
    api_key="ROBOFLOW_API_KEY",
    model_configs=model_configs,
)

# run inference on a single image
result = base_model.predict("image.jpeg")

print(result)

plot(
    image=cv2.imread("image.jpeg"),
    detections=result,
    classes=base_model.ontology.classes(),
)

# label a folder of images
base_model.label("./context_images", extension=".jpeg")
```

Above, replace:

- `API_KEY`: with your Roboflow API key
- `PROJECT_NAME`: with your Roboflow project ID.
- `VERSION`: with your Roboflow model version.
- `model_type`: with the type of model you want to run. Options are `object-detection`, `classification`, or `segmentation`. This value must be the same as the model type trained on Roboflow Universe.

You can run multiple models on a single image. This is ideal if you need to identify multiple objects using different models hosted on Roboflow Universe. To run multiple models, add the models you want to run in the `model_configs` list. For example:

```python
model_configs = [
    ("PROJECT_ID", VERSION_NUMBER),
    ("PROJECT_ID", VERSION_NUMBER)
]
```

All models will be run on every image.

[Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).
[Learn how to retrieve a model ID](https://docs.roboflow.com/api-reference/workspace-and-project-ids).

## License

This project is licensed under an [MIT license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
