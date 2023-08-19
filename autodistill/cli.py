import json
import os

import click
import cv2
import roboflow
import supervision as sv

from autodistill.detection.caption_ontology import CaptionOntology
from autodistill.registry import import_requisite_module

SUPPORTED_ROBOFLOW_MODEL_UPLOADS = ["yolov5", "yolov5-seg", "yolov8", "yolov8-seg"]

SUPPORTED_MODEL_TYPES = ["detection", "segmentation" "classification"]


@click.command()
@click.argument("images")
@click.option("--base", default="grounding_dino")
@click.option("--target", default="yolov8")
@click.option("--model_type", default="detection")
@click.option("--ontology", default={}, required=True)
@click.option("--epochs", default=200, required=True)
@click.option("--output", default="./dataset", required=True)
@click.option("--upload-to-roboflow", default=False, required=False)
@click.option("--project_license", default="MIT", required=False)
def main(
    images, base, target, model_type, ontology, epochs, output, upload_to_roboflow, project_license
):
    if model_type not in SUPPORTED_MODEL_TYPES:
        print(
            "Model type not supported. Please choose from the following model types: "
            + str(SUPPORTED_MODEL_TYPES)
        )
        exit()

    if ontology == "{}":
        print("No ontology provided. Please provide an ontology.")
        exit()

    print("Loading base model...")
    model = import_requisite_module(base)

    try:
        ontology = json.loads(ontology, strict=False)
    except:
        print("Ontologies must be valid JSON.")
        exit()

    ontology = CaptionOntology(ontology)

    base_model = model(ontology=ontology)

    print("Labeling data...")

    # if dir is file, run label on file
    if os.path.isfile(images):
        classes = ontology.classes()

        box_annotator = sv.BoxAnnotator()

        image = cv2.imread(images)

        base_model.label(
            input_folder=os.path.dirname(images),
            extension=os.path.splitext(images)[1],
            output_folder=output,
        )
        # show results
        detections = base_model.predict(images)

        labels = [
            f"{classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detections
        ]

        annotated_frame = box_annotator.annotate(
            scene=image.copy(),
            detections=detections,
            labels=labels,
        )

        sv.plot_image(annotated_frame, size=(8, 8))

        exit()

    base_model.label(
        input_folder=dir,
        output_folder=output
    )

    print("Loading target model...")
    target_model = import_requisite_module(target)

    print("Training target model...")
    target_model.train(dataset_yaml=os.path.join(output, "data.yaml"), epochs=epochs)

    if upload_to_roboflow:
        roboflow.login()

        rf = roboflow.Roboflow()

        workspace = rf.workspace()

        if model_type == "detection":
            rf_model_value = "object-detection"
        elif model_type == "segmentation":
            rf_model_value = "image-segmentation"
        else:
            rf_model_value = "single-label-classification"

        project = workspace.create_project(
            "autodistill",
            project_license=project_license,
            annotation=ontology.classes()[0],
            project_type=rf_model_value
        )

        rf.workspace().upload_dataset(
            dataset_path=output,
            project_name=project.id.split("/")[-1],
            dataset_format="yolov8",
            project_license=project_license,
            project_type=rf_model_value
        )

        if model_type == "detection":
            model_value = target
        elif model_type == "segmentation":
            model_value = target + "-seg"
        else:
            model_value = target + "-cls"

        if model_value not in SUPPORTED_ROBOFLOW_MODEL_UPLOADS:
            print(
                f"Model type {model_value} is not supported by Roboflow. Please choose one of {SUPPORTED_ROBOFLOW_MODEL_UPLOADS}."
            )
            exit()

        project.generate_version(settings={"augmentation": {}, "preprocessing": {}})

        version = project.versions()[-1].version.split("/")[-1]

        project.version(version).deploy(
            model_type=model_value, model_path=f"./runs/detect/train/"
        )

    print("✨ Your model has been trained! ✨")


if __name__ == "__main__":
    main()
