The `plot()` method allows you to visualize predictions from a detection or segmentation model.

If you use a detection model to run inference (i.e. Grounding DINO), the `plot()` method will plot bounding boxes for each prediction.

If you use a segmentation model to run inference (i.e. Grounded SAM), the `plot()` method will plot segmentation masks for each prediction. 

Here is an example of the method used to annotate predictions from a Grounding DINO model:

=== "Bounding Box"

    ```python
    from autodistill_grounding_dino import GroundingDINO
    from autodistill.detection import CaptionOntology
    from autodistill.utils.plot import plot
    import cv2

    ontology = CaptionOntology(
        {
            "dog": "dog",
        }
    )

    model = GroundingDINO(ontology=ontology)

    result = model.predict("./dog.jpeg")

    plot(
        image=cv2.imread("./dog.jpeg"),
        classes=base_model.ontology.classes(),
        detections=result
    )
    ```

    ![Bounding Box](https://media.roboflow.com/autodistill/plot-bbox.png)

=== "Segmentation Mask"

    ```python
    from autodistill_grounded_sam import GroundedSAM
    from autodistill.detection import CaptionOntology
    from autodistill.utils import plot
    import cv2

    ontology = CaptionOntology(
        {
            "dog": "dog",
        }
    )

    model = GroundedSAM(ontology=ontology)

    result = model.predict("./dog.jpeg")

    plot(
        image=cv2.imread("./dog.jpeg"),
        classes=model.ontology.classes(),
        detections=result
    )
    ```

    ![Segmentation Mask](https://media.roboflow.com/autodistill/plot-mask.png)

## Code Reference

:::autodistill.utils.plot