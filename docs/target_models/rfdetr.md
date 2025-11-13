<span class="cls-button">Object Detection</span>
<span class="sm-button">Segmentation</span>
<span class="tm-button">Target Model</span>

# What is RF-DETR?

[RF-DETR](https://github.com/roboflow/rf-detr) is a real-time, transformer-based object detection and segmentation model developed by Roboflow and released under the Apache 2.0 license. The model architecture is discussed in the "[RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/abs/2511.09554)" paper.

RF-DETR-N outperforms YOLO11-N by 10 mAP points on the Microsoft COCO benchmark while running faster at inference. On RF100-VL, RF-DETR achieves state-of-the-art results, with RF-DETR-M beating YOLO11-M by an average of 5 mAP points across aerial datasets including drone, satellite, and radar.

You can train both RF-DETR object detection and segmentation models with Autodistill.

## Installation

To use the RF-DETR target model, you will need to install the following dependency:

```bash
pip3 install autodistill-rfdetr
```

## Getting Started

To use this package, you will need a dataset in the Microsoft COCO Segmentation data format.

## Quickstart (Object Detection)

```python
from autodistill_rfdetr import RFDETRBase

target_model = RFDETRBase()

# train a model
target_model.train("./labeled_data/", epochs=100)

# run inference on the new model
pred = target_model.predict("./labeled_data/train/images/dog-7.jpg", conf=0.01)
```

## Quickstart (Instance Segmentation)

```python
from autodistill_rfdetr import RFDETRSegPreview

target_model = RFDETRSegPreview()

target_model.train("./labeled_data/data.yaml", epochs=100)

# run inference on the new model
pred = target_model.predict("./labeled_data/train/images/dog-7.jpg", conf=0.01)
```
