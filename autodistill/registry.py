import importlib
import os

AUTODISTILL_MODULES = [
    ("grounded_sam", "GroundedSAM"),
    ("grounding_dino", "GroundingDINO"),
    ("yolov8", "YOLOv8", "yolov8n.pt"),
    ("yolov5", "YOLOv5", "yolov5n.pt"),
    ("fastsam", "FastSAM"),
    ("owl-vit", "OWLViT"),
    ("albef", "ALBEF"),
    ("detic", "DETIC"),
    ("blipv2", "BLIPv2"),
    ("sam-clip", "SAMCLIP"),
    ("dinov2", "DINOv2"),
    ("yolonas", "YOLONAS"),
    ("blip", "BLIP"),
    ("vit", "ViT"),
    ("detr", "DETR"),
]

PACKAGE_NAMES = [i[0] for i in AUTODISTILL_MODULES]


def is_module_installed(module_name):
    try:
        importlib.import_module("autodistill_" + module_name)
    except:
        return False

    return True


def import_requisite_module(module_name):
    if module_name not in PACKAGE_NAMES:
        print(
            f"Module {module_name} not found. Please choose from the following modules: {PACKAGE_NAMES}"
        )
        exit()

    if not is_module_installed(module_name):
        consent = input(
            f"Module {module_name} is not installed. Would you like to install it? (y/n): "
        )
        if consent == "y":
            os.system(f"pip install autodistill_{module_name}")
        else:
            print(
                f"{module_name} is required to run this script with your current configuration. Change your chosen model or run `autodistill` again to install {module_name}."
            )
            exit()

    module = importlib.import_module("autodistill_" + module_name)

    full_module = AUTODISTILL_MODULES[PACKAGE_NAMES.index(module_name)]

    if len(full_module) == 3:
        return getattr(module, full_module[1])(full_module[2])

    return getattr(module, full_module[1])
