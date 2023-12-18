Autodistill lets you use large, foundation vision models to auto-label data for and train small, fine-tuned vision models. This process is called _distillation_.

Your fine-tuned model will run smaller and faster, and will thus be more suitable for deployment on edge devices.

## How Autodistill Works

There are two main concepts in Autodistill:

- A *base model*, which is used to auto-label data. Examples include Grounding DINO, Grounded SAM, and CLIP.
- A *target model*, which is trained on the auto-labeled data. Examples include YOLOv5, YOLOv8, and DETR.

You can use Autodistill with only a base model if you want to label data and run your own training.

You can also use Autodistill with both a base model and a target model to build an end-to-end labeling and training pipeline.

## Distill a Model (Tutorial)

!!! tip

    See the [demo Notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-auto-train-yolov8-model-with-autodistill.ipynb) for a quick introduction to `autodistill`. This notebook walks through building a milk container detection model with no labeling.

    If you want to skip directly to the full code, without the tutorial, go to the [Code Summary](#code-summary) section.

Let's distill a model to see how Autodistill works. We will use Autodistill to auto-label a milk bottle dataset.

For this example, we'll show how to distill [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) into a small [YOLOv8](https://github.com/ultralytics/ultralytics) model using [autodistill-grounded-sam](https://github.com/autodistill/autodistill-grounded-sam) and [autodistill-yolov8](https://github.com/autodistill/autodistill-yolov8).

### Step #1: Install Autodistill and Models

First, install the required dependencies:

```
pip install autodistill autodistill-grounded-sam autodistill-yolov8
```

!!! tip

    [See the Autodistill Supported Models](/supported-models/) list for a list of all supported models.

### Step #2: Set an Ontology

Every base model needs an ontology. An ontology tells Autodistill what you want to identify and what labels should be called in your dataset.

For example, if you want to identify milk bottles, you could use the following ontology:

```python
{
    "milk bottle": "bottle",
    "milk bottle cap": "bottle cap"
}
```

This ontology will tell Autodistill to identify milk bottles and milk bottle caps, and to save the labels as `bottle` and `bottle cap` in your dataset.

### Step #3: Set up the Model

Let's set up our model. Create a new Python file and add the following lines of code:

```python
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
base_model = GroundedSAM(ontology=CaptionOntology({"milk bottle": "bottle", "milk bottle cap": "bottle cap"}))
```

### Step #4: Test the Base Model

We can test our base model using the `predict` function:

```python
results = base_model.predict("milk.jpg")

plot(
    image=cv2.imread("milk.jpg"),
    classes=base_model.ontology.classes(),
    detections=results
)
```

### Step #5: Label a Dataset

Now that we have a base model, we can use it to label a dataset. You can label a dataset using the following code:

```python
base_model.label_folder(
    input_folder="./images",
    output_folder="./labeled-images"
)
```

### Step #6: Train a Target Model

We can use a target model like YOLOv8 to train a model on our labeled dataset. You can train a target model using the following code:

```python
target_model = YOLOv8("yolov8n.pt")
target_model.train("./labeled-images/data.yaml", epochs=200)
```

Your model weights will be saved in a folder called `runs`.

For YOLOv8 models, you can then run inference locally using the [ultralytics](https://github.com/ultralytics/ultralytics) Python package, or deploy your model to Roboflow.

## Code Summary

Here is all of the code we used above, summarized into a single code snippet:

```python
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations

base_model = GroundedSAM(ontology=CaptionOntology({"milk bottle": "bottle", "milk bottle cap": "bottle cap"}))

results = base_model.predict("milk.jpg")

base_model.label_folder(
    input_folder="./images",
    output_folder="./labeled-images"
)

target_model = YOLOv8("yolov8n.pt")
target_model.train("./labeled-images/data.yaml", epochs=200)
```

## Next Steps

Above, we used Autodistill to label a dataset. Next, explore the Autodistill ecosystem of models following our guidance in the [Which model should I use?](/which-model-should-i-use/) guide. This site contains documentation for all Autodistill models, as well as utilities that you can use to work with each model.