With so many base models to use in labeling images, you may wonder "what model should I use for labeling?"

## Detection and Segmentation

We recommend using [Grounding DINO]() as a starting point for detection, and [Grounded SAM]() for segmentation.

Grounding DINO is an effective zero-shot object detector that can identify a wide range of objects, from cars to vinyl record covers.

Grounded SAM combines SAM with Grounding DINO to generate segmentation masks from Grounding DINO predictions.

If Grounding DINO does not identify the object you want to label, consider experimenting with [DETIC](), which can identify over 20,000 classes of objects. DETIC supports an open vocabulary, so you can provide arbitrary text labels for objects.

## Classification

We recommend using [CLIP]() as a starting point for classification, which is effective at classifying a wide range of objects. Read the [CLIP abstract](https://openai.com/research/clip) from OpenAI to learn more.

## Roboflow Universe Models

You can use any of the [50,000+ pre-trained models on Roboflow Universe](https://universe.roboflow.com) to auto-label data. Universe covers an extensive range of models, covering areas from logistics to agriculture.

See the [`autodistill-roboflow-universe`](/base_models/roboflow_universe) base model for more information.

## Understanding Other Models

The guidance above is a starting point, but there are many other models from which you can choose.

Below is a list of all supported models not covered above, as well as notes about their usage.

Some models may no longer be recommended because a new model surpasses its performance.

### Detection

- LLaVA-1.5: LLaVA 1.5 has significant memory requirements compared to other models. It may generalize well to a wide range of objects due to its language grounding, but more experimentation is needed.
- Kosmos-2: Kosmos-2, like LLaVA-1.5, has significant memory requirements compared to other models.
- OWL-ViT: We recommend using OWLv2 over OWL-ViT.
- CoDet: CoDet is a promising zero-shot detection model which we encourage you to try if Grounding DINO does not identify the objects you want to label.
- VLPart: VLPart is a promising zero-shot detection model which we encourage you to try if Grounding DINO does not identify the objects you want to label.

### Classification

- FastViT: FastViT can identify the classes in the ImageNet 1k dataset. FastViT has fast inference times, which makes its use ideal in applications where inference speed is critical.
- AltCLIP: AltCLIP reports strong zero-shot classification performance in English and Chinese when evaluated against the ImageNet dataset. This model may be useful if you want to provide Chinese prompts to auto-label images.
- DINOv2: An embedding model that may be useful for zero-shot classification.
- MetaCLIP: MetaCLIP is an open source CLIP model. It may be worth experimenting with if OpenAI's CLIP model does not perform well on your dataset.
- BLIP: BLIP is a zero-shot classifier. It has higher memory requirements than CLIP, but may perform better on some datasets.
- ALBEF: ALBEF is a zero-shot classifier. It has higher memory requirements than CLIP, but may perform better on some datasets.