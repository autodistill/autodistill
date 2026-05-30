import json
import sys
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

from autodistill.core.embedding_model import EmbeddingModel
from autodistill.detection.caption_ontology import CaptionOntology
from autodistill.detection.detection_base_model import DetectionBaseModel
from autodistill.detection.hard_negative_miner import HardNegativeMiner


class FakeOntology:
    def __init__(self, classes=None):
        self._classes = classes or ["target"]

    def classes(self):
        return self._classes


class FakeDetections:
    def __init__(self, class_id=None, confidence=None):
        self.class_id = np.asarray(class_id) if class_id is not None else None
        self.confidence = (
            np.asarray(confidence, dtype=float) if confidence is not None else None
        )

        if confidence is not None:
            self._length = len(confidence)
        elif class_id is not None:
            self._length = len(class_id)
        else:
            self._length = 0

    def __len__(self):
        return self._length


class FakeBaseModel:
    def __init__(self, predictions=None, classes=None):
        self.predictions = predictions or {}
        self.ontology = FakeOntology(classes=classes)

    def predict(self, input):
        return self.predictions.get(Path(input).name, FakeDetections())


class FakeEmbedder:
    def __init__(self, image_embeddings, text_embedding):
        self.image_embeddings = image_embeddings
        self.text_embedding = text_embedding
        self.embed_text_calls = []

    def embed_text(self, input):
        self.embed_text_calls.append(input)
        return self.text_embedding

    def embed_image(self, input):
        embedding = self.image_embeddings[Path(input).name]
        if isinstance(embedding, Exception):
            raise embedding

        return embedding


class InferenceModeCheckingEmbedder(FakeEmbedder):
    def __init__(self, image_embeddings, text_embedding, torch_module):
        super().__init__(image_embeddings, text_embedding)
        self.torch_module = torch_module

    def embed_text(self, input):
        assert self.torch_module.inference_active is True
        return super().embed_text(input)

    def embed_image(self, input):
        assert self.torch_module.inference_active is True
        return super().embed_image(input)


class ConcreteColorEmbedder(EmbeddingModel):
    def embed_text(self, input):
        return np.array([1.0, 0.0, 0.0])

    def embed_image(self, input):
        with Image.open(input) as image:
            rgb = image.convert("RGB").resize((1, 1))
            return np.asarray(rgb, dtype=float).reshape(-1) / 255.0


class ConcreteDetectionBaseModel(DetectionBaseModel):
    def predict(self, input):
        if Path(input).name == "true_positive.jpg":
            return sv.Detections(
                xyxy=np.array([[0.0, 0.0, 1.0, 1.0]]),
                confidence=np.array([0.95]),
                class_id=np.array([0]),
            )

        return sv.Detections(
            xyxy=np.empty((0, 4)),
            confidence=np.array([]),
            class_id=np.array([], dtype=int),
        )


def write_image(path, color=(255, 255, 255)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def make_dataset(tmp_path, train_count=10):
    dataset_dir = tmp_path / "dataset"
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    valid_images = dataset_dir / "valid" / "images"
    valid_labels = dataset_dir / "valid" / "labels"

    for directory in (train_images, train_labels, valid_images, valid_labels):
        directory.mkdir(parents=True, exist_ok=True)

    (dataset_dir / "data.yaml").write_text(
        "train: train/images\nval: valid/images\nnc: 1\nnames: ['target']\n",
        encoding="utf-8",
    )

    for index in range(train_count):
        write_image(train_images / f"train_{index}.jpg")
        (train_labels / f"train_{index}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )

    write_image(valid_images / "valid_0.jpg")
    (valid_labels / "valid_0.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    return dataset_dir


def test_mines_nearest_negative_and_writes_empty_label(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    write_image(candidate_pool / "near.jpg")
    write_image(candidate_pool / "far.jpg")

    embedder = FakeEmbedder(
        image_embeddings={
            "near.jpg": np.array([1.0, 0.0]),
            "far.jpg": np.array([0.0, 1.0]),
        },
        text_embedding=np.array([1.0, 0.0]),
    )
    miner = HardNegativeMiner(embedder=embedder, base_model=FakeBaseModel())

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target prompt",
        max_ratio=0.2,
        top_k=1,
    )

    assert embedder.embed_text_calls == ["target prompt"]
    assert len(report.accepted) == 1
    assert report.accepted[0].source_path.endswith("near.jpg")

    destination_image = Path(report.accepted[0].destination_image_path)
    destination_label = Path(report.accepted[0].destination_label_path)
    assert destination_image.is_file()
    assert destination_label.is_file()
    assert destination_label.read_text(encoding="utf-8") == ""

    original_label = dataset_dir / "train" / "labels" / "train_0.txt"
    assert original_label.read_text(encoding="utf-8") == "0 0.5 0.5 0.25 0.25\n"
    assert (dataset_dir / "valid" / "images" / "valid_0.jpg").is_file()

    manifest_lines = (
        (dataset_dir / "hard_negative_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    manifest = [json.loads(line) for line in manifest_lines]
    assert manifest[0]["accepted"] is True
    assert manifest[0]["reason"] == "accepted"
    assert manifest[0]["error"] is None


def test_rejects_target_detection_and_continues_to_next_candidate(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    write_image(candidate_pool / "contains_target.jpg")
    write_image(candidate_pool / "lookalike_negative.jpg")

    embedder = FakeEmbedder(
        image_embeddings={
            "contains_target.jpg": np.array([1.0, 0.0]),
            "lookalike_negative.jpg": np.array([0.9, 0.1]),
        },
        text_embedding=np.array([1.0, 0.0]),
    )
    base_model = FakeBaseModel(
        predictions={
            "contains_target.jpg": FakeDetections(class_id=[0], confidence=[0.95]),
            "lookalike_negative.jpg": FakeDetections(),
        }
    )
    miner = HardNegativeMiner(embedder=embedder, base_model=base_model)

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        max_ratio=0.2,
        top_k=1,
        confirm_threshold=0.05,
    )

    assert len(report.accepted) == 1
    assert report.accepted[0].source_path.endswith("lookalike_negative.jpg")
    assert len(report.rejected) == 1
    assert report.rejected[0].source_path.endswith("contains_target.jpg")
    assert report.rejected[0].reason == "target_detected"
    assert report.rejected[0].max_target_confidence == 0.95


def test_rejects_real_supervision_target_detection(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    write_image(candidate_pool / "contains_target.jpg")

    embedder = FakeEmbedder(
        image_embeddings={"contains_target.jpg": np.array([1.0, 0.0])},
        text_embedding=np.array([1.0, 0.0]),
    )
    detections = sv.Detections(
        xyxy=np.array([[0.0, 0.0, 1.0, 1.0]]),
        confidence=np.array([0.95]),
        class_id=np.array([0]),
    )
    base_model = FakeBaseModel(predictions={"contains_target.jpg": detections})
    miner = HardNegativeMiner(embedder=embedder, base_model=base_model)

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        max_ratio=0.2,
        confirm_threshold=0.05,
    )

    assert len(report.accepted) == 0
    assert len(report.rejected) == 1
    assert report.rejected[0].reason == "target_detected"
    assert report.rejected[0].max_target_confidence == 0.95


def test_mines_with_concrete_model_subclasses_and_real_detections(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    write_image(candidate_pool / "near_negative.jpg", color=(240, 30, 30))
    write_image(candidate_pool / "far_negative.jpg", color=(0, 0, 255))
    write_image(candidate_pool / "true_positive.jpg", color=(255, 0, 0))
    (candidate_pool / "corrupt.jpg").write_text("not an image", encoding="utf-8")

    ontology = CaptionOntology({"red target": "target"})
    miner = HardNegativeMiner(
        embedder=ConcreteColorEmbedder(ontology=ontology),
        base_model=ConcreteDetectionBaseModel(ontology=ontology),
    )

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="red target",
        target_class="target",
        max_ratio=0.2,
        confirm_threshold=0.05,
    )

    assert len(report.accepted) == 2
    assert {Path(decision.source_path).name for decision in report.accepted} == {
        "near_negative.jpg",
        "far_negative.jpg",
    }
    assert {decision.reason for decision in report.rejected} == {
        "embedding_failed",
        "target_detected",
    }

    for decision in report.accepted:
        assert Path(decision.destination_image_path).is_file()
        destination_label = Path(decision.destination_label_path)
        assert destination_label.is_file()
        assert destination_label.read_text(encoding="utf-8") == ""

    rejected_by_source = {
        Path(decision.source_path).name: decision for decision in report.rejected
    }
    assert rejected_by_source["true_positive.jpg"].max_target_confidence == 0.95
    assert "UnidentifiedImageError" in rejected_by_source["corrupt.jpg"].error

    manifest = [
        json.loads(line)
        for line in (dataset_dir / "hard_negative_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(manifest) == 4
    assert sum(decision["accepted"] for decision in manifest) == 2


def test_confirm_threshold_zero_accepts_no_detection_candidate(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    write_image(candidate_pool / "negative.jpg")

    embedder = FakeEmbedder(
        image_embeddings={"negative.jpg": np.array([1.0, 0.0])},
        text_embedding=np.array([1.0, 0.0]),
    )
    miner = HardNegativeMiner(embedder=embedder, base_model=FakeBaseModel())

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        max_ratio=0.2,
        confirm_threshold=0,
    )

    assert len(report.accepted) == 1
    assert len(report.rejected) == 0


def test_respects_max_ratio(tmp_path):
    dataset_dir = make_dataset(tmp_path, train_count=10)
    candidate_pool = tmp_path / "candidates"
    image_embeddings = {}

    for index in range(5):
        file_name = f"candidate_{index}.jpg"
        write_image(candidate_pool / file_name)
        image_embeddings[file_name] = np.array([1.0, index / 10.0])

    embedder = FakeEmbedder(
        image_embeddings=image_embeddings,
        text_embedding=np.array([1.0, 0.0]),
    )
    miner = HardNegativeMiner(embedder=embedder, base_model=FakeBaseModel())

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        max_ratio=0.2,
    )

    assert len(report.accepted) == 2
    hard_negative_labels = list(
        (dataset_dir / "train" / "labels").glob("hard_negative_*.txt")
    )
    assert len(hard_negative_labels) == 2


def test_target_class_filters_non_target_detections(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    write_image(candidate_pool / "other_class.jpg")

    embedder = FakeEmbedder(
        image_embeddings={"other_class.jpg": np.array([1.0, 0.0])},
        text_embedding=np.array([1.0, 0.0]),
    )
    base_model = FakeBaseModel(
        predictions={
            "other_class.jpg": FakeDetections(class_id=[1], confidence=[0.99])
        },
        classes=["target", "other"],
    )
    miner = HardNegativeMiner(embedder=embedder, base_model=base_model)

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        target_class="target",
        max_ratio=0.2,
    )

    assert len(report.accepted) == 1
    assert report.accepted[0].max_target_confidence == 0.0


def test_dry_run_does_not_copy_files(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    write_image(candidate_pool / "candidate.jpg")

    embedder = FakeEmbedder(
        image_embeddings={"candidate.jpg": np.array([1.0, 0.0])},
        text_embedding=np.array([1.0, 0.0]),
    )
    miner = HardNegativeMiner(embedder=embedder, base_model=FakeBaseModel())

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        max_ratio=0.2,
        dry_run=True,
    )

    assert len(report.accepted) == 1
    assert report.accepted[0].destination_image_path is None
    assert list((dataset_dir / "train" / "images").glob("hard_negative_*.jpg")) == []


def test_embedding_failure_is_recorded_and_mining_continues(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    write_image(candidate_pool / "bad_embedding.jpg")
    write_image(candidate_pool / "good.jpg")

    embedder = FakeEmbedder(
        image_embeddings={
            "bad_embedding.jpg": ValueError("bad vector"),
            "good.jpg": np.array([1.0, 0.0]),
        },
        text_embedding=np.array([1.0, 0.0]),
    )
    miner = HardNegativeMiner(embedder=embedder, base_model=FakeBaseModel())

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        max_ratio=0.2,
        top_k=1,
    )

    assert len(report.accepted) == 1
    assert report.accepted[0].source_path.endswith("good.jpg")
    assert len(report.rejected) == 1
    assert report.rejected[0].reason == "embedding_failed"
    assert report.rejected[0].error == "ValueError: bad vector"

    manifest = [
        json.loads(line)
        for line in (dataset_dir / "hard_negative_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(decision["reason"] == "embedding_failed" for decision in manifest)


def test_embedding_calls_use_optional_torch_inference_context(tmp_path, monkeypatch):
    class FakeInferenceMode:
        def __enter__(self):
            FakeTorch.inference_active = True

        def __exit__(self, exc_type, exc_value, traceback):
            FakeTorch.inference_active = False

    class FakeTorch:
        inference_active = False

        @staticmethod
        def inference_mode():
            return FakeInferenceMode()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    write_image(candidate_pool / "candidate.jpg")

    embedder = InferenceModeCheckingEmbedder(
        image_embeddings={"candidate.jpg": np.array([1.0, 0.0])},
        text_embedding=np.array([1.0, 0.0]),
        torch_module=FakeTorch,
    )
    miner = HardNegativeMiner(embedder=embedder, base_model=FakeBaseModel())

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        max_ratio=0.2,
        top_k=1,
    )

    assert len(report.accepted) == 1
    assert FakeTorch.inference_active is False


def test_copy_failure_is_recorded_and_mining_continues(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    bad_image = candidate_pool / "bad_image.jpg"
    bad_image.parent.mkdir(parents=True, exist_ok=True)
    bad_image.write_text("not an image", encoding="utf-8")
    write_image(candidate_pool / "good.jpg")

    embedder = FakeEmbedder(
        image_embeddings={
            "bad_image.jpg": np.array([1.0, 0.0]),
            "good.jpg": np.array([0.9, 0.1]),
        },
        text_embedding=np.array([1.0, 0.0]),
    )
    miner = HardNegativeMiner(embedder=embedder, base_model=FakeBaseModel())

    report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        max_ratio=0.2,
        top_k=1,
    )

    assert len(report.accepted) == 1
    assert report.accepted[0].source_path.endswith("good.jpg")
    assert len(report.rejected) == 1
    assert report.rejected[0].source_path.endswith("bad_image.jpg")
    assert report.rejected[0].reason == "copy_failed"
    assert "UnidentifiedImageError" in report.rejected[0].error


def test_repeated_mining_does_not_overwrite_existing_background(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    candidate_pool = tmp_path / "candidates"
    write_image(candidate_pool / "candidate.jpg")

    embedder = FakeEmbedder(
        image_embeddings={"candidate.jpg": np.array([1.0, 0.0])},
        text_embedding=np.array([1.0, 0.0]),
    )
    miner = HardNegativeMiner(embedder=embedder, base_model=FakeBaseModel())

    first_report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        max_ratio=0.2,
        top_k=1,
    )
    first_label = Path(first_report.accepted[0].destination_label_path)
    first_label.write_text("sentinel", encoding="utf-8")

    second_report = miner.mine(
        dataset_dir=str(dataset_dir),
        candidate_pool=str(candidate_pool),
        target_prompt="target",
        max_ratio=0.2,
        top_k=1,
    )

    second_label = Path(second_report.accepted[0].destination_label_path)
    assert second_label != first_label
    assert first_label.read_text(encoding="utf-8") == "sentinel"
    assert second_label.read_text(encoding="utf-8") == ""

    manifest_lines = (
        (dataset_dir / "hard_negative_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(manifest_lines) == 2
