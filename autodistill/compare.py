from autodistill.plot import plot
import cv2
import supervision as sv

from typing import List

def compare(models: list, images: List[str]):
    image_results = []
    model_results = []
    
    for model in models:
        # get model class name
        model_name = model.__class__.__name__

        for image in images:
            results = model.predict(image)

            image_data = cv2.imread(image)

            image_result = plot(image_data, results, classes=model.ontology.prompts(), raw=True)

            image_results.append(image_result)

            model_results.append(model_name)

    sv.plot_images_grid(image_results, grid_size=(len(models), len(images)), titles=model_results, size=(16, 16))
