<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Segment Anything HQ Module

This repository contains the code supporting the Segment Anything base model for use with [Autodistill](https://github.com/autodistill/autodistill).

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

This project is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!