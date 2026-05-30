from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from autodistill.core.embedding_model import EmbeddingModel

from .detection_base_model import DetectionBaseModel

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class HardNegativeDecision:
    """
    A decision made for one hard-negative candidate.

    Attributes:
        source_path: The candidate image path.
        destination_image_path: The copied training image path, if accepted.
        destination_label_path: The empty YOLO label path, if accepted.
        similarity: Cosine similarity to the target prototype.
        accepted: Whether the candidate was added to the dataset.
        reason: The reason the candidate was accepted or rejected.
        max_target_confidence: Maximum base-model target confidence, if available.
        error: Error details for candidates that failed during mining.
    """

    source_path: str
    destination_image_path: str | None
    destination_label_path: str | None
    similarity: float | None
    accepted: bool
    reason: str
    max_target_confidence: float | None
    error: str | None = None


@dataclass
class HardNegativeMiningReport:
    """
    Summary of a hard-negative mining run.

    Attributes:
        accepted: Candidate decisions that were added to the dataset.
        rejected: Candidate decisions that were evaluated but not added.
    """

    accepted: List[HardNegativeDecision]
    rejected: List[HardNegativeDecision]

    @property
    def decisions(self) -> List[HardNegativeDecision]:
        """
        Return accepted and rejected decisions in one list.
        """

        return self.accepted + self.rejected


class HardNegativeMiner:
    """
    Mine look-alike negative images for an Autodistill detection dataset.

    The miner ranks candidate images by embedding similarity to a target prompt,
    confirms that the base detection model does not find the target, and appends
    accepted candidates to the training split as YOLO background images with
    empty label files.

    Args:
        embedder: An embedding model that implements `embed_image()` and
            `embed_text()`.
        base_model: A detection base model used to confirm candidates are
            negatives.
    """

    def __init__(self, embedder: EmbeddingModel, base_model: DetectionBaseModel):
        self.embedder = embedder
        self.base_model = base_model

    def mine(
        self,
        dataset_dir: str,
        candidate_pool: str,
        target_prompt: str,
        target_class: str | None = None,
        split: str = "train",
        max_ratio: float = 0.10,
        top_k: int | None = None,
        confirm_threshold: float = 0.05,
        min_similarity: float | None = None,
        extensions: Tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
        output_manifest: str | None = None,
        allow_unscored_detections: bool = False,
        dry_run: bool = False,
    ) -> HardNegativeMiningReport:
        """
        Select and append hard-negative background images.

        Args:
            dataset_dir: Autodistill detection dataset directory containing
                `data.yaml` and split folders.
            candidate_pool: Directory of unlabeled candidate images.
            target_prompt: Text prompt used to build the target embedding.
            target_class: Optional dataset class name to use when filtering base
                model detections. If omitted for a one-class ontology, the only
                class is used. If omitted for a multi-class ontology, all
                detections are treated as target detections.
            split: Dataset split to append mined negatives to. Defaults to
                `train`.
            max_ratio: Maximum mined negatives as a fraction of existing split
                images.
            top_k: Optional upper bound on the number of mined negatives.
            confirm_threshold: Reject candidates when the base model detects the
                target at or above this confidence.
            min_similarity: Optional minimum cosine similarity for candidates.
            extensions: Candidate image extensions to include.
            output_manifest: Optional JSONL path for decisions. Defaults to
                `<dataset_dir>/hard_negative_manifest.jsonl`.
            allow_unscored_detections: If false, reject candidates with target
                detections that do not expose confidence values.
            dry_run: If true, rank and confirm candidates but do not copy files.

        Returns:
            A `HardNegativeMiningReport` with accepted and rejected decisions.
        """

        dataset_path = Path(dataset_dir)
        candidate_path = Path(candidate_pool)
        split_images_dir, split_labels_dir = self._validate_dataset(dataset_path, split)

        if not candidate_path.is_dir():
            raise ValueError(f"Candidate pool does not exist: {candidate_pool}")
        if max_ratio <= 0:
            raise ValueError("max_ratio must be greater than 0")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0 when provided")
        if confirm_threshold < 0:
            raise ValueError("confirm_threshold must be greater than or equal to 0")

        existing_count = len(self._list_images(split_images_dir, extensions))
        if existing_count == 0:
            raise ValueError(f"No images found in {split_images_dir}")

        limit = math.floor(existing_count * max_ratio)
        if top_k is not None:
            limit = min(limit, top_k)

        if limit <= 0:
            report = HardNegativeMiningReport(accepted=[], rejected=[])
            self._write_manifest(report, dataset_path, output_manifest)
            return report

        candidates = self._candidate_images(candidate_path, dataset_path, extensions)

        with self._embedding_inference_context():
            prototype = self._flatten_embedding(self.embedder.embed_text(target_prompt))
            ranked_candidates, rejected = self._rank_candidates(candidates, prototype)

        target_class_id = self._resolve_target_class_id(target_class)

        accepted: List[HardNegativeDecision] = []

        for source_path, similarity in ranked_candidates:
            if min_similarity is not None and similarity < min_similarity:
                rejected.append(
                    self._rejected_decision(
                        source_path=source_path,
                        similarity=similarity,
                        reason="below_min_similarity",
                    )
                )
                continue

            try:
                is_negative, max_confidence, reason = self._confirm_negative(
                    source_path=source_path,
                    target_class_id=target_class_id,
                    confirm_threshold=confirm_threshold,
                    allow_unscored_detections=allow_unscored_detections,
                )
            except Exception as error:
                rejected.append(
                    self._rejected_decision(
                        source_path=source_path,
                        similarity=similarity,
                        reason="confirm_failed",
                        error=error,
                    )
                )
                continue

            if not is_negative:
                rejected.append(
                    self._rejected_decision(
                        source_path=source_path,
                        similarity=similarity,
                        reason=reason,
                        max_target_confidence=max_confidence,
                    )
                )
                continue

            destination_image_path = None
            destination_label_path = None
            if not dry_run:
                try:
                    (
                        destination_image_path,
                        destination_label_path,
                    ) = self._append_background(
                        source_path=source_path,
                        split_images_dir=split_images_dir,
                        split_labels_dir=split_labels_dir,
                    )
                except Exception as error:
                    rejected.append(
                        self._rejected_decision(
                            source_path=source_path,
                            similarity=similarity,
                            reason="copy_failed",
                            max_target_confidence=max_confidence,
                            error=error,
                        )
                    )
                    continue

            accepted.append(
                HardNegativeDecision(
                    source_path=str(source_path),
                    destination_image_path=str(destination_image_path)
                    if destination_image_path
                    else None,
                    destination_label_path=str(destination_label_path)
                    if destination_label_path
                    else None,
                    similarity=similarity,
                    accepted=True,
                    reason="accepted",
                    max_target_confidence=max_confidence,
                )
            )

            if len(accepted) >= limit:
                break

        report = HardNegativeMiningReport(accepted=accepted, rejected=rejected)
        self._write_manifest(report, dataset_path, output_manifest)
        return report

    def _validate_dataset(self, dataset_path: Path, split: str) -> Tuple[Path, Path]:
        if not (dataset_path / "data.yaml").is_file():
            raise ValueError(
                f"Autodistill dataset is missing data.yaml: {dataset_path}"
            )

        split_images_dir = dataset_path / split / "images"
        split_labels_dir = dataset_path / split / "labels"

        if not split_images_dir.is_dir():
            raise ValueError(
                f"Dataset split image directory missing: {split_images_dir}"
            )
        if not split_labels_dir.is_dir():
            raise ValueError(
                f"Dataset split label directory missing: {split_labels_dir}"
            )

        return split_images_dir, split_labels_dir

    def _candidate_images(
        self, candidate_pool: Path, dataset_path: Path, extensions: Tuple[str, ...]
    ) -> List[Path]:
        dataset_root = dataset_path.resolve()
        candidates = []

        for path in self._list_images(candidate_pool, extensions):
            resolved_path = path.resolve()
            if self._is_relative_to(resolved_path, dataset_root):
                continue
            candidates.append(path)

        return candidates

    def _rank_candidates(
        self, candidates: List[Path], prototype: np.ndarray
    ) -> Tuple[List[Tuple[Path, float]], List[HardNegativeDecision]]:
        ranked = []
        rejected = []

        for candidate in candidates:
            try:
                embedding = self._flatten_embedding(
                    self.embedder.embed_image(str(candidate))
                )
            except Exception as error:
                rejected.append(
                    self._rejected_decision(
                        source_path=candidate,
                        similarity=None,
                        reason="embedding_failed",
                        error=error,
                    )
                )
                continue

            similarity = self._cosine_similarity(prototype, embedding)
            ranked.append((candidate, similarity))

        return sorted(ranked, key=lambda item: item[1], reverse=True), rejected

    def _confirm_negative(
        self,
        source_path: Path,
        target_class_id: int | None,
        confirm_threshold: float,
        allow_unscored_detections: bool,
    ) -> Tuple[bool, float | None, str]:
        detections = self.base_model.predict(str(source_path))
        max_confidence, reason = self._max_target_confidence(
            detections, target_class_id
        )

        if reason in ("no_detections", "no_target_detections"):
            return True, max_confidence, "accepted"

        if max_confidence is None:
            if allow_unscored_detections:
                return True, None, "accepted_unscored_target_detections"
            return False, None, reason

        if max_confidence >= confirm_threshold:
            return False, max_confidence, "target_detected"

        return True, max_confidence, "accepted"

    def _max_target_confidence(self, detections, target_class_id: int | None):
        if len(detections) == 0:
            return 0.0, "no_detections"

        detection_count = len(detections)
        target_mask = np.ones(detection_count, dtype=bool)
        class_ids = getattr(detections, "class_id", None)

        if target_class_id is not None and class_ids is not None:
            target_mask = np.asarray(class_ids) == target_class_id

        if not np.any(target_mask):
            return 0.0, "no_target_detections"

        confidences = getattr(detections, "confidence", None)
        if confidences is None:
            return None, "unscored_target_detections"

        target_confidences = np.asarray(confidences, dtype=float)[target_mask]
        if target_confidences.size == 0:
            return 0.0, "no_target_detections"

        return float(np.max(target_confidences)), "target_confidence"

    def _resolve_target_class_id(self, target_class: str | None) -> int | None:
        classes = self.base_model.ontology.classes()

        if target_class is None:
            if len(classes) == 1:
                return 0
            return None

        if target_class not in classes:
            raise ValueError(
                f"target_class must be one of {classes}; got {target_class}"
            )

        return classes.index(target_class)

    def _append_background(
        self, source_path: Path, split_images_dir: Path, split_labels_dir: Path
    ) -> Tuple[Path, Path]:
        destination_stem = self._destination_stem(source_path)
        destination_image_path, destination_label_path = self._available_destination(
            destination_stem, split_images_dir, split_labels_dir
        )

        try:
            with Image.open(source_path) as image:
                image.convert("RGB").save(destination_image_path, format="JPEG")

            destination_label_path.write_text("", encoding="utf-8")
        except Exception:
            if destination_image_path.exists():
                destination_image_path.unlink()
            if destination_label_path.exists():
                destination_label_path.unlink()
            raise

        return destination_image_path, destination_label_path

    def _available_destination(
        self, stem: str, split_images_dir: Path, split_labels_dir: Path
    ) -> Tuple[Path, Path]:
        index = 0

        while True:
            suffix = f"_{index}" if index > 0 else ""
            destination_image_path = split_images_dir / f"{stem}{suffix}.jpg"
            destination_label_path = split_labels_dir / f"{stem}{suffix}.txt"

            if (
                not destination_image_path.exists()
                and not destination_label_path.exists()
            ):
                return destination_image_path, destination_label_path

            index += 1

    def _destination_stem(self, source_path: Path) -> str:
        digest = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()
        source_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_path.stem)
        return f"hard_negative_{digest[:10]}_{source_stem}"

    def _list_images(self, directory: Path, extensions: Tuple[str, ...]) -> List[Path]:
        normalized_extensions = tuple(extension.lower() for extension in extensions)
        images = []

        for root, _, files in os.walk(directory):
            for file_name in files:
                if file_name.lower().endswith(normalized_extensions):
                    images.append(Path(root) / file_name)

        return sorted(images)

    def _write_manifest(
        self,
        report: HardNegativeMiningReport,
        dataset_path: Path,
        output_manifest: str | None,
    ) -> None:
        manifest_path = (
            Path(output_manifest)
            if output_manifest is not None
            else dataset_path / "hard_negative_manifest.jsonl"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with manifest_path.open("a", encoding="utf-8") as manifest:
            for decision in report.decisions:
                manifest.write(json.dumps(asdict(decision)) + "\n")

    def _rejected_decision(
        self,
        source_path: Path,
        similarity: float | None,
        reason: str,
        max_target_confidence: float | None = None,
        error: Exception | None = None,
    ) -> HardNegativeDecision:
        return HardNegativeDecision(
            source_path=str(source_path),
            destination_image_path=None,
            destination_label_path=None,
            similarity=similarity,
            accepted=False,
            reason=reason,
            max_target_confidence=max_target_confidence,
            error=self._format_error(error) if error is not None else None,
        )

    def _format_error(self, error: Exception) -> str:
        return f"{type(error).__name__}: {error}"

    def _flatten_embedding(self, embedding) -> np.ndarray:
        return np.asarray(embedding, dtype=float).reshape(-1)

    def _cosine_similarity(self, first: np.ndarray, second: np.ndarray) -> float:
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator == 0:
            return 0.0

        return float(np.dot(first, second) / denominator)

    def _embedding_inference_context(self):
        try:
            import torch
        except ImportError:
            return nullcontext()

        inference_mode = getattr(torch, "inference_mode", None)
        if inference_mode is not None:
            return inference_mode()

        no_grad = getattr(torch, "no_grad", None)
        if no_grad is not None:
            return no_grad()

        return nullcontext()

    def _is_relative_to(self, path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False

        return True
