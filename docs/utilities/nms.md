You can apply Non-Maximum Suppression (NMS) to predictions from a detection model to remove overlapping bounding boxes.

To do so, add `.with_nms()` to the result of any `predict()` or `predict_sahi()` method from an object detection model.

Here is an example of running NMS on predictions from a Grounding DINO model:

=== "Without NMS"

    ```
    import cv2
    import supervision as sv

    from autodistill_grounding_dino import GroundingDINO
    from autodistill.detection import CaptionOntology

    base_model = GroundingDINO(ontology=CaptionOntology({"person": "person"}))

    detections = base_model.predict_sahi("./image.jpg").with_nms()

    plot(
        image=cv2.imread("./image.jpg"),
        detections=detections.with_nms()
    )
    ```

    ![Without NMS](https://media.roboflow.com/autodistill/without-nms.png)

=== "With NMS"

    ```
    import cv2
    import supervision as sv

    from autodistill_grounding_dino import GroundingDINO
    from autodistill.detection import CaptionOntology

    base_model = GroundingDINO(ontology=CaptionOntology({"person": "person"}))

    detections = base_model.predict_sahi("./image.jpg").with_nms()

    plot(
        image=cv2.imread("./image.jpg"),
        detections=detections.with_nms()
    )
    ```

    ![With NMS](https://media.roboflow.com/autodistill/with-nms.png)