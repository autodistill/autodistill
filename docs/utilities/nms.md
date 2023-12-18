You can apply Non-Maximum Suppression (NMS) to predictions from a detection model to remove overlapping bounding boxes.

To do so, add `.with_nms()` to the result of any `predict()` or `predict_sahi()` method from an object detection model.

Here is an example of running NMS on predictions from a Grounding DINO model:

=== "Without NMS"

    ```python
    from autodistill_owlv2 import OWLv2
    from autodistill.detection import CaptionOntology
    from autodistill.utils import plot

    import cv2

    ontology = CaptionOntology({"person": "person"})

    base_model = OWLv2(ontology=ontology)

    detections = base_model.predict("./dog.jpeg")

    plot(
        image=cv2.imread("./dog.jpeg"),
        detections=detections,
        classes=base_model.ontology.classes(),
    )
    ```

    ![Without NMS](https://media.roboflow.com/autodistill/without-nms.png)

=== "With NMS"

    ```python
    from autodistill_owlv2 import OWLv2
    from autodistill.detection import CaptionOntology
    from autodistill.utils import plot

    import cv2

    ontology = CaptionOntology({"person": "person"})

    base_model = OWLv2(ontology=ontology)

    detections = base_model.predict("./dog.jpeg")

    plot(
        image=cv2.imread("./dog.jpeg"),
        detections=detections.with_nms(),
        classes=base_model.ontology.classes(),
    )
    ```

    ![With NMS](https://media.roboflow.com/autodistill/with-nms.png)