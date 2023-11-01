import os
import random
import shutil

import cv2
import supervision as sv
import tqdm
import yaml
from PIL import Image

VALID_ANNOTATION_TYPES = ["box", "mask"]


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
