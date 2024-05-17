import csv
import json
import os

import click
import cv2
import roboflow
import supervision as sv

from autodistill.detection.caption_ontology import CaptionOntology
from autodistill.registry import import_requisite_module

# load model matrix from models.csv
with open(os.path.join(os.path.dirname(__file__), "models.csv")) as csvfile:
    models = list(csv.DictReader(csvfile))

SUPPORTED_ROBOFLOW_MODEL_UPLOADS = ["yolov5", "yolov5-seg", "yolov8", "yolov8-seg"]

SUPPORTED_MODEL_TYPES = ["detection", "segmentation" "classification"]

SUPPORTED_DATASET_FORMATS = ["yolov8", "yolov5", "voc"]


@click.command()
@click.argument("images")
@click.option("--models", default=False, help="Show available models.")
@click.option(
    "--base", default="grounding_dino", help="Base model to use for labeling images."
)
@click.option("--target", default="yolov8", help="Target model to use for training.")
@click.option(
    "--model_type",
    default="detection",
    help="Type of model to train (detection, segmentation, classification).",
)
@click.option(
    "--ontology", default={}, required=True, help="Ontology to use for labeling images."
)
@click.option("--epochs", default=200, required=True, help="Number of epochs to train.")
@click.option(
    "--output",
    default="./dataset",
    required=True,
    help="Output directory for labeled data.",
)
@click.option(
    "--upload-to-roboflow",
    default=False,
    required=True,
    help="Upload dataset and trained model to Roboflow.",
)
@click.option(
    "--project_name",
    default="autodistill",
    required=False,
    help="Name of Roboflow project.",
)
@click.option(
    "--project_license",
    default="MIT",
    required=False,
    help="License to use for Roboflow dataset. Set to `private` to upload a dataset and model privately.",
)
@click.option(
    "--dataset_format",
    default="voc",
    required=False,
    help="Dataset format to use for Roboflow project (voc, yolov5, yolov8).",
)
@click.option(
    "-y",
    default=False,
    required=False,
    help="Automatically answer yes to all install prompts.",
)
@click.option(
    "--test",
    default=False,
    required=False,
    help="Run a base model on a folder of images (up to 9 images) and show the results.",
)
def main(
    images,
    models,
    base,
    target,
    model_type,
    ontology,
    epochs,
    output,
    upload_to_roboflow,
    project_name,
    project_license,
    dataset_format,
    y,
    test,
):
    if models:
        print("Available models:")
        for model in models:
            print(model["name"])
        exit()

    if dataset_format not in SUPPORTED_DATASET_FORMATS:
        print(
            "Dataset format not supported. Please choose from the following dataset formats: "
            + str(SUPPORTED_DATASET_FORMATS)
        )
        exit()

    if model_type not in SUPPORTED_MODEL_TYPES:
        print(
            "Model type not supported. Please choose from the following model types: "
            + str(SUPPORTED_MODEL_TYPES)
        )
        exit()

    if ontology == "{}":
        print("No ontology provided. Please provide an ontology.")
        exit()

    if upload_to_roboflow:
        roboflow.login()

        if model_type == "detection":
            model_value = target
        elif model_type == "segmentation":
            model_value = target + "-seg"
        else:
            model_value = target + "-cls"

        if model_value not in SUPPORTED_ROBOFLOW_MODEL_UPLOADS:
            print(
                f"Model type {model_value} is not supported for deployment on Roboflow. Please choose one of {SUPPORTED_ROBOFLOW_MODEL_UPLOADS}."
            )
            exit()

    print("Loading base model...")
    model = import_requisite_module(base, noninteractive_install=y)

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
            for _, _, confidence, class_id, _, _ in detections
        ]

        annotated_frame = box_annotator.annotate(
            scene=image.copy(),
            detections=detections,
            labels=labels,
        )

        sv.plot_image(annotated_frame, size=(8, 8))

        print("Saving results to ./result.jpg...")

        cv2.imwrite(os.path.join(os.getcwd(), "result.jpg"), annotated_frame)

        exit()
    elif test:
        files = os.listdir(images)[:9]
        # order alphabetically

        files.sort()

        results = []

        for file in files:
            classes = ontology.classes()

            box_annotator = sv.BoxAnnotator()

            image = cv2.imread(os.path.join(images, file))

            detections = base_model.predict(os.path.join(images, file))

            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _, _ in detections
            ]

            annotated_frame = box_annotator.annotate(
                scene=image.copy(),
                detections=detections,
                labels=labels,
            )

            results.append(annotated_frame)

        sv.plot_images_grid(results, grid_size=(3, 3))

        exit()

    base_model.label(input_folder=images, output_folder=output)

    print("Loading target model...")
    target_model = import_requisite_module(target, noninteractive_install=y)

    print("Training target model...")
    target_model.train(dataset_yaml=os.path.join(output, "data.yaml"), epochs=epochs)

    if upload_to_roboflow:
        rf = roboflow.Roboflow()

        workspace = rf.workspace()

        if model_type == "detection":
            rf_model_value = "object-detection"
        elif model_type == "segmentation":
            rf_model_value = "image-segmentation"
        else:
            rf_model_value = "single-label-classification"

        if project_name not in [project.name for project in workspace.projects()]:
            project = workspace.create_project(
                project_name=project_name,
                project_license=project_license,
                annotation=ontology.classes()[0],
                project_type=rf_model_value,
            )

        rf.workspace().upload_dataset(
            dataset_path=output,
            project_name=project.id.split("/")[-1],
            dataset_format=dataset_format,
            project_license=project_license,
            project_type=rf_model_value,
        )

        if model_type == "detection":
            model_value = target
        elif model_type == "segmentation":
            model_value = target + "-seg"
        else:
            model_value = target + "-cls"

        project.generate_version(settings={"augmentation": {}, "preprocessing": {}})

        version = project.versions()[-1].version.split("/")[-1]

        project.version(version).deploy(
            model_type=model_value, model_path=f"./runs/detect/train/"
        )

    print("✨ Your model has been trained! ✨")


if __name__ == "__main__":
    main()
