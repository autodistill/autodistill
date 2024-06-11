<span class="sm-button">Segmentation</span>
<span class="bm-button">Base Model</span>

# What is Segment Anything HQ?

[SAM HQ](https://github.com/SysCV/sam-hq) is a zero-shot segmentation model capable of producing detailed masks, developed by [ETH VIS](https://github.com/SysCV). SAM HQ can segment an entire image into masks, or use points to segment specific parts of an object. You can use Segment Anything with Autodistill to segment objects. Segment Anything does not assign classes, so you should use SAM HQ model with a tool like Grounding DINO or GPT-4V.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [SAM HQ Autodistill documentation](https://autodistill.github.io/autodistill/base_models/samhq/).

## Installation

To use SAM HQ with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-sam-hq
```

## Quickstart

```python
from autodistill_sam_hq import HQSAM

base_model = HQSAM(None)

masks = base_model.predict("./image.jpeg")

print(masks)
```

## License

This project is licensed under an [Apache 2.0 license](https://github.com/autodistill/autodistill-sam-hq/blob/main/LICENSE).