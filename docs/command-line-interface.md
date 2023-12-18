You can use Autodistill with a command line interface (CLI). 

The CLI allows you to run inference on a model or auto-label a folder of imageswihout writing a labeling script.

## Installation

To install the CLI, install the Autodistill Python package:

```bash
pip install autodistill
```

The CLI accepts several arguments:

- `images`: The path to the folder of images you want to label.
- `--base`: The base model you want to use for labeling. This can be any model from the [Autodistill Model Zoo](/model-zoo).
- `--target`: The target model you want to use to train a model with your labeled dataset.
- `--ontology`: The ontology you want to use for labeling. This must be a mapping of text prompts to send to a model to the label you want to save in your dataset. For example, `{"acoustic guitar": "guitar"}` will send the text prompt `acoustic guitar` to the model, then save the label as `guitar` in your dataset.
- `--output`: The path to the folder where you want to save your labeled dataset.

Here is an example:

```bash
autodistill images --base="grounding_dino" --target="yolov8" --ontology '{"prompt": "label"}' --output="./dataset"
```

This command will label all images in a directory called `images` with Grounding DINO and use the labeled images to train a YOLOv8 model. Grounding DINO will label all images with the "prompt" and save the label as the "label". You can specify as many prompts and labels as you want. The resulting dataset will be saved in a folder called `dataset`.