[Slicing Aided Hyper Inference](https://github.com/obss/sahi) (SAHI) is a technique that improves the detection rate of small objects in an image. SAHI involves splitting up an image into segments, then runs inference on each segment. Then, the results from each segment are combined into a single result.

Because SAHI runs inference on separate segments, it will take longer to run inference on an image with SAHI than without SAHI.

You can use SAHI when running inference on a single image with Autodistill, or when using Autodistill to label a folder of images.

## Use SAHI in a Single Prediction

To use SAHI in a single prediction, use the `sahi` parameter in the `predict()` method:

```python
import cv2
import supervision as sv

from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

base_model = GroundingDINO(ontology=CaptionOntology({"person": "person"}))

detections = base_model.predict_sahi("./image.jpg")

classes = ["person"]

box_annotator = sv.BoxAnnotator()

labels = [
	f"{classes[class_id]} {confidence:0.2f}"
	for _, _, confidence, class_id, _
	in detections
]

image = cv2.imread("./image.jpg")

annotated_frame = box_annotator.annotate(
	scene=image.copy(),
	detections=detections,
	labels=labels
)

sv.plot_image(image=annotated_frame, size=(16, 16))
```

Here are the results before and after SAHI:

=== "Without SAHI"

    ![Without SAHI](https://media.roboflow.com/autodistill/without-sahi.png)

=== "With SAHI"

    ![With SAHI](https://media.roboflow.com/autodistill/sahi.png)

The image processed with SAHI detected more people.

## Use SAHI to Label a Folder of Images

To use SAHI to label a folder of images, use the `sahi` parameter in the `label()` method on any base model:

```python
base_model.label_folder(
    input_folder="./images",
    output_folder="./labeled-images",
    sahi=True
)
```

## See Also

- [Using SAHI with supervision](https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/)