import importlib
import os

AUTODISTILL_MODULES = [
    "groundedsam",
    "yolov8",
    "yolov5"
]

def is_module_installed(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    else:
        return True

def import_requisite_module(module_name):
    if module_name not in AUTODISTILL_MODULES:
        raise ValueError(f"Module {module_name} not found.")
    
    if not is_module_installed("autodistill_" + module_name):
        os.system("pip install autodistill_" + module_name)

    if module_name == "groundedsam":
        from autodistill_grounded_sam import GroundedSAM

        return GroundedSAM
    elif module_name == "yolov8":
        from autodistill_yolov8 import YOLOv8

        return YOLOv8
    elif module_name == "yolov5":
        from autodistill_yolov5 import YOLOv5

        return YOLOv5