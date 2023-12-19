Our goal is for `autodistill` to support using all foundation models as Base Models and most SOTA supervised models as Target Models. We focused on object detection and segmentation
tasks first but plan to launch classification support soon! In the future, we hope `autodistill` will also be used for models beyond computer vision.

* ✅ - complete (click row/column header to go to repo)
* 🚧 - work in progress

### object detection

| base / target | [YOLOv8](https://github.com/autodistill/autodistill-yolov8) | [YOLO-NAS](https://github.com/autodistill/autodistill-yolonas) | [YOLOv5](https://github.com/autodistill/autodistill-yolov5) | [DETR](https://github.com/autodistill/autodistill-detr) | YOLOv6 | YOLOv7 | MT-YOLOv6 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [DETIC](https://github.com/autodistill/autodistill-detic) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [GroundedSAM](https://github.com/autodistill/autodistill-grounded-sam) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [GroundingDINO](https://github.com/autodistill/autodistill-grounding-dino) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [OWL-ViT](https://github.com/autodistill/autodistill-owl-vit) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [SAM-CLIP](https://github.com/autodistill/autodistill-sam-clip) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [LLaVA-1.5](https://github.com/autodistill/autodistill-llava) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [Kosmos-2](https://github.com/autodistill/autodistill-kosmos-2) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [OWLv2](https://github.com/autodistill/autodistill-owlv2) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [Roboflow Universe Models (50k+ pre-trained models)](https://github.com/autodistill/autodistill-roboflow-universe) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [CoDet](https://github.com/autodistill/autodistill-codet) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [VLPart](https://github.com/autodistill/autodistill-vlpart) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [Azure Custom Vision](https://github.com/autodistill/autodistill-azure-vision) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [AWS Rekognition](https://github.com/autodistill/autodistill-rekognition) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |
| [Google Vision](https://github.com/autodistill/autodistill-gcp-vision) | ✅ | ✅ | ✅ | ✅ | 🚧 |  |  |


### instance segmentation

| base / target | [YOLOv8](https://github.com/autodistill/autodistill-yolov8) | [YOLO-NAS](https://github.com/autodistill/autodistill-yolonas) | [YOLOv5](https://github.com/autodistill/autodistill-yolov5) | YOLOv7 | Segformer |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [GroundedSAM](https://github.com/autodistill/autodistill-grounded-sam) | ✅ | 🚧 | 🚧 |  |  |
| SAM-CLIP | ✅ | 🚧 | 🚧 |  |  |
| SegGPT | ✅ | 🚧 | 🚧 |  |  |
| FastSAM | 🚧 | 🚧 | 🚧 |  |  |


### classification

| base / target | [ViT](https://github.com/autodistill/autodistill-vit) | [YOLOv8](https://github.com/autodistill/autodistill-yolov8) | [YOLOv5](https://github.com/autodistill/autodistill-yolov5) |
|:---:|:---:|:---:|:---:|
| [CLIP](https://github.com/autodistill/autodistill-clip) | ✅ | ✅ | 🚧 |
| [MetaCLIP](https://github.com/autodistill/autodistill-metaclip) | ✅ | ✅ | 🚧 |
| [DINOv2](https://github.com/autodistill/autodistill-dinov2) | ✅ | ✅ | 🚧 |
| [BLIP](https://github.com/autodistill/autodistill-blip) | ✅ | ✅ | 🚧 |
| [ALBEF](https://github.com/autodistill/autodistill-albef) | ✅ | ✅ | 🚧 |
| [FastViT](https://github.com/autodistill/autodistill-fastvit) | ✅ | ✅ | 🚧 |
| [AltCLIP](https://github.com/autodistill/autodistill-altcip) | ✅ | ✅ | 🚧 |
| Fuyu | 🚧 | 🚧 | 🚧 |
| Open Flamingo | 🚧 | 🚧 | 🚧 |
| GPT-4 |  |  |  |
| PaLM-2 |  |  |  |


## Roboflow Model Deployment Support

You can optionally deploy some Target Models trained using Autodistill on Roboflow. Deploying on Roboflow allows you to use a range of concise SDKs for using your model on the edge, from [roboflow.js](https://docs.roboflow.com/inference/web-browser) for web deployment to [NVIDIA Jetson](https://docs.roboflow.com/inference/nvidia-jetson) devices.

The following Autodistill Target Models are supported by Roboflow for deployment:

| model name | Supported? |
|:---:|:---:|
| YOLOv8 Object Detection | ✅ |
| YOLOv8 Instance Segmentation | ✅ |
| YOLOv5 Object Detection | ✅ |
| YOLOv5 Instance Segmentation | ✅ |
| YOLOv8 Classification |  |