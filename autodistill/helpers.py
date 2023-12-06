import os
import random
import shutil
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import requests
import supervision as sv
import tqdm
import yaml
from PIL import Image

VALID_ANNOTATION_TYPES = ["box", "mask"]
ACCEPTED_RETURN_FORMATS = ["PIL", "cv2", "numpy"]


def load_image(
    image: Any,
    return_format="cv2",
) -> Any:
    """
    Load an image from a file path, URI, PIL image, or numpy array.

    This function is for use by Autodistill modules. You don't need to use it directly.

    Args:
        image: The image to load
        return_format: The format to return the image in

    Returns:
        The image in the specified format
    """
    if return_format not in ACCEPTED_RETURN_FORMATS:
        raise ValueError(f"return_format must be one of {ACCEPTED_RETURN_FORMATS}")

    if isinstance(image, Image.Image) and return_format == "PIL":
        return image
    elif isinstance(image, Image.Image) and return_format == "cv2":
        # channels need to be reversed for cv2
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, Image.Image) and return_format == "numpy":
        return np.array(image)

    if isinstance(image, np.ndarray) and return_format == "PIL":
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, np.ndarray) and return_format == "cv2":
        return image
    elif isinstance(image, np.ndarray) and return_format == "numpy":
        return image

    if isinstance(image, str) and image.startswith("http"):
        if return_format == "PIL":
            response = requests.get(image)
            return Image.open(BytesIO(response.content))
        elif return_format == "cv2" or return_format == "numpy":
            response = requests.get(image)
            pil_image = Image.open(BytesIO(response.content))
            return np.array(pil_image)
    elif os.path.isfile(image):
        if return_format == "PIL":
            return Image.open(image)
        elif return_format == "cv2":
            # channels need to be reversed for cv2
            return cv2.cvtColor(np.array(Image.open(image)), cv2.COLOR_RGB2BGR)
        elif return_format == "numpy":
            pil_image = Image.open(image)
            return np.array(pil_image)
    else:
        raise ValueError(f"{image} is not a valid file path or URI")


def split_data(base_dir, split_ratio=0.8):
    images_dir = os.path.join(base_dir, "images")
    annotations_dir = os.path.join(base_dir, "annotations")

    # Correct the image file names if they have an extra dot before the extension
    for file in os.listdir(images_dir):
        if file.count(".") > 1:
            new_file_name = file.replace("..", ".")
            os.rename(
                os.path.join(images_dir, file), os.path.join(images_dir, new_file_name)
            )

    # Convert .png and .jpeg images to .jpg
    for file in os.listdir(images_dir):
        if file.endswith(".png"):
            img = Image.open(os.path.join(images_dir, file))
            rgb_img = img.convert("RGB")
            rgb_img.save(os.path.join(images_dir, file.replace(".png", ".jpg")))
            os.remove(os.path.join(images_dir, file))
        if file.endswith(".jpeg"):
            img = Image.open(os.path.join(images_dir, file))
            rgb_img = img.convert("RGB")
            rgb_img.save(os.path.join(images_dir, file.replace(".jpeg", ".jpg")))
            os.remove(os.path.join(images_dir, file))

    # Get list of all files (removing the image file extension)
    all_files = os.listdir(images_dir)
    all_files = [os.path.splitext(f)[0] for f in all_files if f.endswith(".jpg")]

    # Shuffle the files
    random.shuffle(all_files)

    # Compute the splitting index
    split_idx = int(len(all_files) * split_ratio)

    # Split the files
    train_files = all_files[:split_idx]
    valid_files = all_files[split_idx:]

    # Make directories for train and valid
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Make images and labels subdirectories
    train_images_dir = os.path.join(train_dir, "images")
    train_labels_dir = os.path.join(train_dir, "labels")
    valid_images_dir = os.path.join(valid_dir, "images")
    valid_labels_dir = os.path.join(valid_dir, "labels")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    # Move the files
    for file in train_files:
        shutil.move(os.path.join(images_dir, file + ".jpg"), train_images_dir)
        shutil.move(os.path.join(annotations_dir, file + ".txt"), train_labels_dir)

    for file in valid_files:
        shutil.move(os.path.join(images_dir, file + ".jpg"), valid_images_dir)
        shutil.move(os.path.join(annotations_dir, file + ".txt"), valid_labels_dir)

    # Load the existing YAML file to get the names
    with open(os.path.join(base_dir, "data.yaml"), "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        names = data["names"]

    # Rewrite the YAML file
    with open(os.path.join(base_dir, "data.yaml"), "w") as file:
        data = {
            "train": os.path.abspath(base_dir) + "/train/images",
            "val": os.path.abspath(base_dir) + "/valid/images",
            "nc": len(names),
            "names": names,
        }
        yaml.dump(data, file)


def split_video_frames(video_path: str, output_dir: str, stride: int) -> None:
    """
    Split a video into frames and save them to a directory.

    Args:
        video_path: The path to the video
        output_dir: The directory to save the frames to
        stride: The stride to use when splitting the video into frames

    Returns:
        None
    """
    video_paths = sv.list_files_with_extensions(
        directory=video_path, extensions=["mov", "mp4", "MOV", "MP4"]
    )

    for name in tqdm(video_paths):
        image_name_pattern = name + "-{:05d}.jpg"
        with sv.ImageSink(
            target_dir_path=output_dir, image_name_pattern=image_name_pattern
        ) as sink:
            for image in sv.get_video_frames_generator(
                source_path=str(video_path), stride=stride
            ):
                sink.save_image(image=image)
