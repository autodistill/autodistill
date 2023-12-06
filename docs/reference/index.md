This section of the Autodistill documentation covers the low-level Autodistill API. This section may be useful for advanced users who want to understand the Autodisitll API in more depth.

## Autodistill API, in a Nutshell

The Autodistill API consists of three concepts:

1. Base models, which are used to auto-label data;
2. Target models, which are trained on the auto-labeled data; and
3. Ontologies, which tell Autodistill what you want to identify and what labels should be called in your dataset.

There are three different kinds of base models:

1. Detection models, which identify objects in images and return bounding boxes and/or segmentation masks;
2. Classification models, which classify images and return a class label; and;
3. Embedding models, which return semantic embeddings for images. These are used with the `EmbeddingOntology` class.

The base model you use depends on the type of data you want to label. You can also combine models using the [ComposedDetectionModel](/utilities/combine-models) API, which allows you to refine labels from detection models.

There are two different kinds of target models:

1. Detection models, and;
2. Classification models.

Autodistill does not support training embedding models.