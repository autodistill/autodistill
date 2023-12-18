Autodistill has not been optimized for labeling large datasets, but this work is in progress. In the mean time, we recommend labeling only a few hundred images at a time, to the maximum of how many images you can store in memory.

## How the Autodistill labeling process works

During image labeling, a data structure is built that contains:

1. A numpy representation of an image;
2. The labels for the image, and;
3. The image file name.

If you are labeling large datasets, this data structure will get large. For example, if you have 10,000 images in a folder to label, this data structure will contain 10,000 images. This can cause memory issues if your system doesn't have enough memory to store all images.

We are working on a system that will prevent the need to store images in memory during the labeling process. This system will also include an intelligent label resumption system, so if labeling stops for any reason you will be able to resume labeling from where you stopped.

Follow [Issue #93 in the Autodistill GitHub repository](https://github.com/autodistill/autodistill/pull/93).