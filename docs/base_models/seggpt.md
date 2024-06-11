<span class="sm-button">Segmentation</span>
<span class="bm-button">Base Model</span>

# What is SegGPT?

[SegGPT](https://github.com/baaivision/Painter/tree/main/SegGPT) is a transformer-based, few-shot semantic segmentation model developed by [BAAI Vision](https://github.com/baaivision).

This model performs well on task-specific segmentation tasks when given a few labeled images from which to learn features about the objects you want to identify.

## Installation

To use SegGPT with Autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-seggpt
```

## About SegGPT

SegGPT performs "in-context" segmentation. This means it requires a handful of pre-labelled "context" images.

You will need some labeled images to use SegGPT. Don't have any labeled images? Check out [Roboflow Annotate](https://roboflow.com/annotate), a feature-rich annotation tool from which you can export data for use with Autodistill.

## Quickstart

```python
from autodistill_seggpt import SegGPT, FewShotOntology

base_model = SegGPT(
    ontology=FewShotOntology(supervision_dataset)
)

base_model.label("./unlabelled-climbing-photos", extension=".jpg")
```

## How to load data from Roboflow

Labelling and importing images is easy!

You can use [Roboflow Annotate](https://roboflow.com/annotate) to label a few images (5-10 should work fine). For your Project Type, make sure to pick Instance Segmentation, as you will be labelling with polygons.

Once you have labelled your images, you can press Generate > Generate New Version. You can use all the default options--no Augmentations are necessary.

Once your dataset version is generated, you can press Export > Continue.

Then you will get some download code to copy. It should look something like this:

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="ABCDEFG")
project = rf.workspace("lorem-ipsum").project("dolor-sit-amet")
dataset = project.version(1).download("yolov8")
```

Note: if you are not using a notebook environment, you should remove `!pip install roboflow` from your code, and run `pip install roboflow` in your terminal instead.

To import your dataset into Autodistill, run the following:

```py
import supervision as sv

supervision_dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=f"{dataset.location}/train/images",
    annotations_directory_path=f"{dataset.location}/train/labels",
    data_yaml_path=f"{dataset.location}/data.yaml"
)
```
