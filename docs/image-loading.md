All Autodistill base models (i.e. Grounding DINO or CLIP) support providing a file name and loading the corresponding image for use in labeling. Some models also enable passing images directly from the following formats:

- PIL `Image`
- cv2 image
- URL, from which an image is retrieved
- A file name, which is loaded as an image

This is handled by the low-level `load_image` function. This function allows you to pass any of the above formats. The PIL and cv2 formats are ideal if you already have an image in memory. Base models use this function to request the format the model needs. If a model needs an image in a format different from what you have provided -- for example, if you provided a file name and the model needs a PIL `Image` object -- the `load_image` function will convert the image to the correct format.

The following models support the `load_image` function. The `PIL` and `cv2` states to what format `load_image` will convert your image (if necessary) to pass your image into a model.

- AltCLIP: PIL
- CLIP: PIL
- Grounding DINO: cv2
- MetaCLIP: PIL
- RemoteCLIP: PIL
- Transformers: PIL
- SAM HQ: cv2
- Segment Anything: cv2
- DETIC: PIL
- VLPart: PIL
- CoDet: PIL
- OWLv2: PIL
- FastViT: PIL
- FastSAM: cv2
- SegGPT: PIL
- OWLViT: PIL
- BLIPv2: PIL
- DINOv2: PIL
- Grounded SAM: cv2
- BLIP: PIL

## `load_image` function

:::autodistill.helpers.load_image