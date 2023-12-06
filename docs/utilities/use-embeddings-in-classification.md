You can use embeddings in an `EmbeddingOntology` to classify images and detections with Autodistill.

This has two uses:

1. Classify entire images using embeddings computed with an embedding model, and;
2. Classify regions of an image using the [ComposedDetectionModel]() API.

This API is especially useful if a classification model with embedding support (i.e. CLIP) struggles with a text prompt you provide.

Consider a scenario where you want to classify vinyl records. You could compute an embedding for each album cover, then use those embeddings for classification.

There are two `EmbeddingOntology` classes:

- `EmbeddingOntologyImage`: Accepts a mapping from a text prompt (the class you will use in labeling) to an embedding. The embedding model you use for labeling (i.e. CLIP) will automatically compute embeddings for each image.
- `EmbeddingOntologyRaw`: Accepts a mapping from a text prompt (the class you will use in labeling) to an embedding. You must compute the embeddings yourself, then provide them to the `EmbeddingOntologyRaw` class.

In most cases, `EmbeddingOntologyImage` is the best choice, because Autodistill handles loading the model.

However, if you already have embeddings, `EmbeddingOntologyRaw` is a better choice.

If you use `EmbeddingOntologyImage` with pre-computed embeddings, you must use the same embedding model as the model you use for classification in Autodistill, otherwise auto-labeling will return inaccurate results.

## EmbeddingOntologyImage Example

In the example below, Grounding DINO is used to detect album covers, then CLIP is used to classify the album covers.

Six images are provided as references in the `EmbeddingOntologyImage` class. These images are embedded by CLIP, then used for classification for each vinyl record detected by Grounding DINO.

[Learn more about the ComposedDetectionModel API](/utilities/combine-models).

```python
from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from autodistill.core import EmbeddingOntologyImage
from autodistill.core.combined_detection_model import CombinedDetectionModel

import torch
import clip
from PIL import Image
import os
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

INPUT_FOLDER = "samples"
DATASET_INPUT = "./images"
DATASET_OUTPUT = "./dataset"
PROMPT = "album cover"

images = os.listdir("samples")

images_to_classes = {
    "midnights": "IMG_9022.jpeg",
    "men amongst mountains": "323601467684.jpeg",
    "we are": "IMG_9056.jpeg",
    "oh wonder": "Images (5).jpeg",
    "brightside": "Images (4).jpeg",
    "tears for fears": "Images (3).jpeg"
}

model = CombinedDetectionModel(
    detection_model=GroundingDINO(
        CaptionOntology({PROMPT: PROMPT})
    ),
    classification_model=CLIP(
        EmbeddingOntologyImage(images_to_classes)
    )
)

result = model.predict("./images/example.jpeg")

plot(
    image=cv2.imread("./images/example.jpeg"),
    detections=result
)
```

Here is the result from inference:

![EmbeddingOntologyImage Example](https://media.roboflow.com/autodistill/annotation.png)

The album cover is annotated with the label "men amougst mountains".

Grounding DINO successfully identified an album cover, then our EmbeddingOntologyImage classified the album cover.

## EmbeddingOntologyRaw Example

In the example below, we load embeddings from a file, where embeddings are in the form `{prompt: embedding}`.

```python
from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from autodistill.core import EmbeddingOntology
from autodistill.core.custom_detection_model import CustomDetectionModel

import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

INPUT_FOLDER = "samples"
DATASET_INPUT = "./images"
DATASET_OUTPUT = "./dataset"
PROMPT = "album cover"

images = os.listdir("samples")

with open("embeddings.json", "r") as f:
    classes_to_embeddings = json.load(f)

SAMCLIP = CustomDetectionModel(
    detection_model=GroundingDINO(
        CaptionOntology({PROMPT: PROMPT})
    ),
    classification_model=CLIP(
        EmbeddingOntology(classes_to_embeddings.items())
    )
)

result = model.predict("./images/example.jpeg")

plot(
    image=cv2.imread("./images/example.jpeg"),
    detections=result
)
```

## Code Reference

:::autodistill.core.embedding_ontology