<span class="tc-button">Text Classification</span>
<span class="bm-button">Target Model</span>

# What is DistilBERT?

DistilBERT is a languae model architecture commonly used in training sentence classification models. You can use `autodistill` to train a DistilBERT model that classifies text.

## Installation

To use the DistilBERT target model, you will need to install the following dependency:

```bash
pip3 install autodistill-distilbert-text
```

## Quickstart

The DistilBERT module takes in `.jsonl` files and trains a text classification model.

Each record in the JSONL file should have an entry called `text` that contains the text to be classified. The `label` entry should contain the ground truth label for the text. This format is returned by Autodistill base text classification models like the GPTClassifier.

Here is an example entry of a record used to train a research paper subject classifier:

```json
{"title": "CC-GPX: Extracting High-Quality Annotated Geospatial Data from Common Crawl", "content": "arXiv:2405.11039v1 Announce Type: new \nAbstract: The Common Crawl (CC) corpus....", "classification": "natural language processing"}
```

```python
from autodistill_distilbert import DistilBERT

target_model = DistilBERT()

# train a model
target_model.train("./data.jsonl", epochs=200)

# run inference on the new model
pred = target_model.predict("Geospatial data.", conf=0.01)

print(pred)
# geospatial
```
