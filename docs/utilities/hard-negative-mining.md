# Mine Hard Negatives

Hard-negative mining adds look-alike background images to an object detection
dataset before training a target model. This can help reduce false positives
when a distilled model sees deployment images that look similar to the target
class but do not contain it.

`HardNegativeMiner` works through the dataset artifact created by Autodistill:

1. Build a text embedding for the target prompt.
2. Embed an unlabeled candidate pool.
3. Rank candidate images by similarity to the target.
4. Use the base detection model to reject candidates that appear to contain the
   target.
5. Append accepted candidates to `train/images` and create empty YOLO label
   files in `train/labels`.
6. Write a JSONL manifest for audit. The manifest records accepted, rejected,
   and failed candidates.

This utility does not add a new embedding model dependency to `autodistill`.
Pass any installed model that implements `embed_image()` and `embed_text()`.

```python
from autodistill.detection import CaptionOntology, HardNegativeMiner
from autodistill_clip import CLIP
from autodistill_grounded_sam import GroundedSAM

ontology = CaptionOntology({"milk bottle": "milk_bottle"})
base_model = GroundedSAM(ontology=ontology)

base_model.label(
    input_folder="./images",
    output_folder="./dataset",
)

embedder = CLIP(ontology=ontology)
miner = HardNegativeMiner(embedder=embedder, base_model=base_model)

report = miner.mine(
    dataset_dir="./dataset",
    candidate_pool="./unlabeled_pool",
    target_prompt="milk bottle",
    target_class="milk_bottle",
    max_ratio=0.10,
    confirm_threshold=0.05,
)

print(f"Added {len(report.accepted)} hard negatives")
```

The default `max_ratio` is `0.10`, so the miner adds at most one hard negative
for every ten existing training images. Validation images are left unchanged.

## Parameters

- `dataset_dir`: Autodistill detection dataset with `data.yaml` and split
  folders.
- `candidate_pool`: Directory of unlabeled images that may contain look-alikes.
- `target_prompt`: Text prompt used to rank candidates by embedding similarity.
- `target_class`: Optional class name used to filter base-model detections.
- `max_ratio`: Maximum number of mined negatives as a fraction of existing
  training images.
- `top_k`: Optional upper bound on mined negatives.
- `confirm_threshold`: Reject candidates when the base model detects the target
  at or above this confidence.
- `min_similarity`: Optional minimum cosine similarity required for a candidate.
- `output_manifest`: Optional append-only JSONL path for accepted, rejected,
  and failed decisions.
- `dry_run`: Rank and confirm candidates without copying images.

## Verifying impact

Evaluate the mined dataset against a baseline before using it in production.
Compare:

- the original Autodistill dataset,
- the dataset plus random background images from the same pool, and
- the dataset plus mined hard negatives.

Track precision, recall, mAP, false positives per negative image, and wall-clock
time or epochs to reach the target precision. The random-background control is
important because it separates the value of background images from the value of
embedding-based selection.
