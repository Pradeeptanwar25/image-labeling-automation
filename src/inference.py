from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.category_context import (
    CategoryContext,
    CategoryContextError,
    get_mapped_label,
    load_label_mapping,
)

from src.safe_io import (
    SafeIOError,
    safe_copy_file,
    safe_remove_file,
    safe_remove_tree,
    safe_write_json,
)


class InferenceError(Exception):
    """Raised when dynamic YOLO inference fails."""

    pass


class DynamicYOLOInference:
    """
    Run cumulative inference for one selected
    folder or nested subfolder.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".jfif",
        ".png",
        ".bmp",
        ".webp",
    }

    def __init__(
        self,
        context: CategoryContext,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        image_size: int = 640,
        device: str = "cpu",
    ):
        self.context = context

        self.model_path = (
            context.best_model_file.resolve()
        )

        self.dataset_root = (
            context.dataset_root.resolve()
        )

        self.source_folder = (
            context.selected_dataset_path.resolve()
        )

        self.output_root = (
            context.prediction_root.resolve()
        )

        self.review_store_file = (
            context.review_store_file.resolve()
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.iou_threshold = float(
            iou_threshold
        )

        self.image_size = int(
            image_size
        )

        self.device = str(
            device
        ).strip()

        self.model = None

        self._validate_configuration()

    # =====================================================
    # CONFIGURATION
    # =====================================================

    def _validate_configuration(
        self,
    ) -> None:
        if not self.model_path.is_file():
            raise InferenceError(
                "Production best.pt not found: "
                f"{self.model_path}"
            )

        if not self.dataset_root.is_dir():
            raise InferenceError(
                "Dataset root not found: "
                f"{self.dataset_root}"
            )

        if not self.source_folder.is_dir():
            raise InferenceError(
                "Selected folder not found: "
                f"{self.source_folder}"
            )

        if not (
            0.0
            <= self.confidence_threshold
            <= 1.0
        ):
            raise InferenceError(
                "confidence_threshold must be "
                "between 0 and 1."
            )

        if not (
            0.0
            <= self.iou_threshold
            <= 1.0
        ):
            raise InferenceError(
                "iou_threshold must be "
                "between 0 and 1."
            )

        if self.image_size <= 0:
            raise InferenceError(
                "image_size must be greater than zero."
            )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.review_store_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =====================================================
    # MODEL
    # =====================================================

    def load_model(
        self,
    ):
        try:
            from ultralytics import YOLO

        except ImportError as exc:
            raise InferenceError(
                "Ultralytics is not installed."
            ) from exc

        try:
            self.model = YOLO(
                str(
                    self.model_path
                )
            )

        except Exception as exc:
            raise InferenceError(
                "Could not load production model: "
                f"{exc}"
            ) from exc

        return self.model

    # =====================================================
    # JSON HELPERS
    # =====================================================

    @staticmethod
    def load_json(
        path: str | Path | None,
    ) -> dict[str, Any] | None:
        if path is None:
            return None

        file_path = Path(
            path
        ).resolve()

        if not file_path.is_file():
            return None

        try:
            data = json.loads(
                file_path.read_text(
                    encoding="utf-8"
                )
            )

        except (
            json.JSONDecodeError,
            OSError,
        ):
            return None

        if not isinstance(
            data,
            dict,
        ):
            return None

        return data

    @staticmethod
    def collect_image_paths(
        json_files: list[Path],
    ) -> set[str]:
        paths: set[str] = set()

        for json_file in json_files:
            data = (
                DynamicYOLOInference.load_json(
                    json_file
                )
            )

            if data is None:
                continue

            images = data.get(
                "images",
                [],
            )

            if not isinstance(
                images,
                list,
            ):
                continue

            for record in images:
                if not isinstance(
                    record,
                    dict,
                ):
                    continue

                image_path = str(
                    record.get(
                        "image_path",
                        "",
                    )
                ).strip()

                if image_path:
                    paths.add(
                        Path(
                            image_path
                        )
                        .as_posix()
                        .casefold()
                    )

        return paths

    # =====================================================
    # EXCLUSION SOURCES
    # =====================================================

    def discover_exclusion_files(
        self,
    ) -> list[Path]:
        """
        Collect manual, combined, reviewed and archived
        annotation files for only the selected folder.
        """

        candidates = [
            self.context.normalized_annotations_file,
            self.context.combined_annotations_file,
            self.context.review_store_file,
            self.context.verified_annotations_file,
            self.context.predictions_file,
        ]

        # Preserve existing S26_2 legacy locations.
        if (
            self.context.selected_folder
            == "S26_2"
        ):
            candidates.extend(
                [
                    (
                        self.context.project_root
                        / "data"
                        / "normalized_annotations"
                        / "s26_2_manual_annotations.json"
                    ),
                    (
                        self.context.project_root
                        / "data"
                        / "combined_annotations"
                        / "s26_2_training_annotations.json"
                    ),
                ]
            )

        archive_root = (
            self.context.prediction_archive_root
        )

        if archive_root.is_dir():
            candidates.extend(
                sorted(
                    archive_root.rglob(
                        "predictions.json"
                    )
                )
            )

            candidates.extend(
                sorted(
                    archive_root.rglob(
                        "review_store.json"
                    )
                )
            )

            candidates.extend(
                sorted(
                    archive_root.rglob(
                        "verified_annotations.json"
                    )
                )
            )

        unique_files = []
        seen_paths = set()

        for candidate in candidates:
            resolved = candidate.resolve()

            key = str(
                resolved
            ).casefold()

            if key in seen_paths:
                continue

            if not resolved.is_file():
                continue

            seen_paths.add(
                key
            )

            unique_files.append(
                resolved
            )

        return unique_files

    # =====================================================
    # SHA-256 IMAGE HASH
    # =====================================================

    @staticmethod
    def calculate_sha256(
        image_path: Path,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Calculate exact content hash for one image.

        The source image is read only.
        It is never modified or deleted.
        """

        digest = hashlib.sha256()

        try:
            with image_path.open(
                "rb"
            ) as file_handle:
                while True:
                    chunk = file_handle.read(
                        chunk_size
                    )

                    if not chunk:
                        break

                    digest.update(
                        chunk
                    )

        except OSError as exc:
            raise InferenceError(
                "Could not calculate SHA-256 for "
                f"{image_path}: {exc}"
            ) from exc

        return digest.hexdigest()


    # =====================================================
    # HASHES OF ALREADY PROCESSED IMAGES
    # =====================================================

    def collect_processed_hashes(
        self,
        excluded_paths: set[str],
    ) -> set[str]:
        """
        Calculate hashes for already manual-labelled,
        reviewed or previously predicted images.

        This allows a duplicate copy with a different
        filename/path to be skipped as well.
        """

        processed_hashes = set()

        for relative_path in sorted(
            excluded_paths
        ):
            source_path = (
                self.dataset_root
                / Path(
                    relative_path
                )
            )

            if not source_path.is_file():
                continue

            try:
                image_hash = (
                    self.calculate_sha256(
                        source_path
                    )
                )

            except InferenceError as exc:
                print(
                    "Processed-image hash warning: "
                    f"{exc}"
                )

                continue

            processed_hashes.add(
                image_hash
            )

        return processed_hashes


    # =====================================================
    # FRESH UNIQUE IMAGES
    # =====================================================

    def discover_fresh_images(
        self,
        exclusion_files: list[Path],
    ) -> dict[str, Any]:
        """
        Discover fresh unique images for inference.

        Skip rules:

        1. Image path already exists in manual/review/
           prediction JSON.

        2. Image has the same SHA-256 as an already
           processed image, even if its filename or
           folder is different.

        3. Image has the same SHA-256 as another fresh
           image discovered during this scan.

        Only one physical file per unique SHA-256 hash
        is sent to the model.
        """

        excluded_paths = (
            self.collect_image_paths(
                exclusion_files
            )
        )

        processed_hashes = (
            self.collect_processed_hashes(
                excluded_paths
            )
        )

        seen_hashes = set(
            processed_hashes
        )

        fresh_images = []

        skipped_images = []

        duplicate_images = []

        hash_errors = []

        hash_to_retained_image = {}

        for image_path in sorted(
            self.source_folder.rglob("*")
        ):
            if not image_path.is_file():
                continue

            if (
                image_path.suffix.casefold()
                not in self.SUPPORTED_EXTENSIONS
            ):
                continue

            relative_path = (
                image_path
                .relative_to(
                    self.dataset_root
                )
                .as_posix()
            )

            normalized_path = (
                relative_path.casefold()
            )

            # ---------------------------------------------
            # PATH-BASED PROCESSED IMAGE SKIP
            # ---------------------------------------------

            if normalized_path in excluded_paths:
                skipped_images.append(
                    image_path
                )

                continue

            # ---------------------------------------------
            # SHA-256 CALCULATION
            # ---------------------------------------------

            try:
                image_hash = (
                    self.calculate_sha256(
                        image_path
                    )
                )

            except InferenceError as exc:
                hash_errors.append(
                    {
                        "image_path": (
                            relative_path
                        ),
                        "error": str(
                            exc
                        ),
                    }
                )

                print(
                    "Image hash failed: "
                    f"{relative_path}: {exc}"
                )

                continue

            # ---------------------------------------------
            # CONTENT-BASED DUPLICATE SKIP
            # ---------------------------------------------

            if image_hash in seen_hashes:
                duplicate_images.append(
                    {
                        "image_path": (
                            relative_path
                        ),
                        "sha256": (
                            image_hash
                        ),
                        "duplicate_of": (
                            hash_to_retained_image.get(
                                image_hash
                            )
                        ),
                        "duplicate_of_processed_image": (
                            image_hash
                            in processed_hashes
                        ),
                    }
                )

                continue

            # ---------------------------------------------
            # RETAIN FIRST UNIQUE COPY
            # ---------------------------------------------

            seen_hashes.add(image_hash)

            hash_to_retained_image[
                image_hash
            ] = relative_path

            fresh_images.append(
                image_path
            )

        return {
            "fresh_images": (
                fresh_images
            ),
            "skipped_images": (
                skipped_images
            ),
            "duplicate_images": (
                duplicate_images
            ),
            "hash_errors": (
                hash_errors
            ),
            "excluded_path_count": len(
                excluded_paths
            ),
            "processed_hash_count": len(
                processed_hashes
            ),
            "unique_fresh_hash_count": len(
                fresh_images
            ),
            "duplicate_hash_count": len(
                duplicate_images
            ),
        }


    # =====================================================
    # ARCHIVE PREVIOUS BATCH
    # =====================================================

    def archive_previous_batch(
        self,
    ) -> Path | None:
        files = [
            self.context.predictions_file,
            self.context.inference_summary_file,
        ]

        preview_directory = (
            self.context.preview_directory
        )

        review_store = (
            self.context.review_store_file
        )

        has_existing_batch = (
            any(
                path.is_file()
                for path in files
            )
            or preview_directory.is_dir()
            or review_store.is_file()
        )

        if not has_existing_batch:
            return None

        timestamp = (
            datetime.now()
            .astimezone()
            .strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        archive_directory = (
            self.context.prediction_archive_root
            / f"batch_{timestamp}"
        )

        archive_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        for source in files:
            if source.is_file():
                try:
                    safe_copy_file(
                        source=source,
                        destination=(
                            archive_directory
                            / source.name
                        ),
                        retries=30,
                    )

                except SafeIOError as exc:
                    raise InferenceError(
                        "Could not safely archive "
                        f"{source}: {exc}"
                    ) from exc

        if preview_directory.is_dir():
            archive_preview_directory = (
                archive_directory
                / "previews"
            )

            archive_preview_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            preview_manifest = []

            for preview_file in sorted(
                preview_directory.rglob("*")
            ):
                if not preview_file.is_file():
                    continue

                relative_preview = (
                    preview_file
                    .relative_to(
                        preview_directory
                    )
                    .as_posix()
                )

                suffix = (
                    preview_file.suffix
                    if preview_file.suffix
                    else ".jpg"
                )

                short_name = (
                    hashlib.sha256(
                        relative_preview.encode(
                            "utf-8"
                        )
                    ).hexdigest()[:20]
                    + suffix.lower()
                )

                archive_preview_file = (
                    archive_preview_directory
                    / short_name
                )

                try:
                    safe_copy_file(
                        source=(
                            preview_file
                        ),
                        destination=(
                            archive_preview_file
                        ),
                        retries=30,
                    )

                    preview_manifest.append(
                        {
                            "original_preview": (
                                relative_preview
                            ),
                            "archived_preview": (
                                short_name
                            ),
                        }
                    )

                except OSError as exc:
                    print(
                        "Preview archive warning: "
                        f"{preview_file.name}: "
                        f"{exc}"
                    )

            try:
                safe_write_json(
                    destination=(
                        archive_preview_directory
                        / "preview_manifest.json"
                    ),
                    payload={
                        "preview_count": len(
                            preview_manifest
                        ),
                        "previews": (
                            preview_manifest
                        ),
                    },
                    retries=30,
                )

            except SafeIOError as exc:
                raise InferenceError(
                    "Could not safely save "
                    "preview archive manifest: "
                    f"{exc}"
                ) from exc

        if review_store.is_file():
            try:
                safe_copy_file(
                    source=(
                        review_store
                    ),
                    destination=(
                        archive_directory
                        / "review_store.json"
                    ),
                    retries=30,
                )

            except SafeIOError as exc:
                raise InferenceError(
                    "Could not safely archive "
                    "review store: "
                    f"{exc}"
                ) from exc

        return archive_directory

    # ======================
    # CLEAR CURRENT WORKING BATCH
    # =====================================================

    def clear_working_batch(
        self,
    ) -> None:
        working_files = [
            self.context.predictions_file,
            self.context.inference_summary_file,
            self.context.review_store_file,
        ]

        for path in working_files:
            if path.is_file():
                try:
                    safe_remove_file(
                        path,
                        retries=20,
                    )

                except SafeIOError as exc:
                    raise InferenceError(
                        "Could not safely clear "
                        f"working file {path}: "
                        f"{exc}"
                    ) from exc

        if (
            self.context
            .preview_directory
            .is_dir()
        ):
            try:
                safe_remove_tree(
                    self.context.preview_directory,
                    retries=20,
                )

            except SafeIOError as exc:
                raise InferenceError(
                    "Could not safely clear "
                    "preview directory: "
                    f"{exc}"
                ) from exc

    # =====================================================
    # BOUNDING BOX
    # =====================================================

    @staticmethod
    def clamp_bbox(
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        image_width: int,
        image_height: int,
    ) -> dict[str, float] | None:
        xmin = max(
            0.0,
            min(
                float(xmin),
                float(image_width),
            ),
        )

        ymin = max(
            0.0,
            min(
                float(ymin),
                float(image_height),
            ),
        )

        xmax = max(
            0.0,
            min(
                float(xmax),
                float(image_width),
            ),
        )

        ymax = max(
            0.0,
            min(
                float(ymax),
                float(image_height),
            ),
        )

        if xmax <= xmin:
            return None

        if ymax <= ymin:
            return None

        return {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }

    # =====================================================
    # PARSE RESULT
    # =====================================================

    def parse_result(
        self,
        result: Any,
        image_path: Path,
    ) -> dict[str, Any]:
        with Image.open(
            image_path
        ) as image:
            (
                image_width,
                image_height,
            ) = image.size

        relative_path = (
            image_path
            .relative_to(
                self.dataset_root
            )
            .as_posix()
        )

        annotations = []

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        names = getattr(
            result,
            "names",
            {},
        )

        if boxes is not None:
            for box in boxes:
                try:
                    coordinates = (
                        box.xyxy[0]
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    confidence = float(
                        box.conf[0]
                        .detach()
                        .cpu()
                        .item()
                    )

                    class_id = int(
                        box.cls[0]
                        .detach()
                        .cpu()
                        .item()
                    )

                except Exception:
                    continue

                if (
                    confidence
                    < self.confidence_threshold
                ):
                    continue

                if len(coordinates) != 4:
                    continue

                bbox = self.clamp_bbox(
                    xmin=coordinates[0],
                    ymin=coordinates[1],
                    xmax=coordinates[2],
                    ymax=coordinates[3],
                    image_width=image_width,
                    image_height=image_height,
                )

                if bbox is None:
                    continue

                if isinstance(
                    names,
                    dict,
                ):
                    class_name = str(
                        names.get(
                            class_id,
                            self.context.mapped_label,
                        )
                    )

                else:
                    try:
                        class_name = str(
                            names[
                                class_id
                            ]
                        )

                    except Exception:
                        class_name = (
                            self.context.mapped_label
                        )

                if (
                    class_name
                    != self.context.mapped_label
                ):
                    continue

                annotations.append(
                    {
                        "class_id": (
                            class_id
                        ),
                        "class_name": (
                            class_name
                        ),
                        "confidence": (
                            confidence
                        ),
                        "xmin": (
                            bbox[
                                "xmin"
                            ]
                        ),
                        "ymin": (
                            bbox[
                                "ymin"
                            ]
                        ),
                        "xmax": (
                            bbox[
                                "xmax"
                            ]
                        ),
                        "ymax": (
                            bbox[
                                "ymax"
                            ]
                        ),
                        "source": (
                            "ai_prediction"
                        ),
                        "verified": False,
                    }
                )

        highest_confidence = max(
            (
                annotation[
                    "confidence"
                ]
                for annotation
                in annotations
            ),
            default=None,
        )

        return {
            "image_path": (
                relative_path
            ),
            "image_name": (
                image_path.name
            ),
            "image_width": (
                image_width
            ),
            "image_height": (
                image_height
            ),
            "prediction_count": len(
                annotations
            ),
            "highest_confidence": (
                highest_confidence
            ),
            "review_required": True,
            "annotations": (
                annotations
            ),
            "status": "predicted",
        }

    # =====================================================
    # PREDICT
    # =====================================================

    def predict_images(
        self,
        image_paths: list[Path],
    ) -> list[dict[str, Any]]:
        if self.model is None:
            self.load_model()

        records = []

        total = len(
            image_paths
        )

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            print(
                f"[{index}/{total}] "
                f"Predicting {image_path.name}"
            )

            relative_path = (
                image_path
                .relative_to(
                    self.dataset_root
                )
                .as_posix()
            )

            try:
                results = (
                    self.model.predict(
                        source=str(
                            image_path
                        ),
                        imgsz=(
                            self.image_size
                        ),
                        conf=(
                            self.confidence_threshold
                        ),
                        iou=(
                            self.iou_threshold
                        ),
                        device=(
                            self.device
                        ),
                        verbose=False,
                        save=False,
                    )
                )

                if results:
                    records.append(
                        self.parse_result(
                            results[0],
                            image_path,
                        )
                    )

                else:
                    records.append(
                        {
                            "image_path": (
                                relative_path
                            ),
                            "image_name": (
                                image_path.name
                            ),
                            "status": (
                                "no_result"
                            ),
                            "annotations": [],
                            "review_required": True,
                        }
                    )

            except Exception as exc:
                records.append(
                    {
                        "image_path": (
                            relative_path
                        ),
                        "image_name": (
                            image_path.name
                        ),
                        "status": (
                            "prediction_error"
                        ),
                        "error": str(
                            exc
                        ),
                        "annotations": [],
                        "review_required": True,
                    }
                )

        return records

    # =====================================================
    # PREVIEW
    # =====================================================

    def create_preview(
        self,
        record: dict[str, Any],
    ) -> Path:
        source_path = (
            self.dataset_root
            / Path(
                record[
                    "image_path"
                ]
            )
        )

        if not source_path.is_file():
            raise InferenceError(
                "Source image not found: "
                f"{source_path}"
            )

        self.context.preview_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        with Image.open(
            source_path
        ) as source:
            preview = (
                source
                .convert("RGB")
                .copy()
            )

        draw = ImageDraw.Draw(
            preview
        )

        line_width = max(
            3,
            int(
                min(
                    preview.size
                )
                / 250
            ),
        )

        color = (
            255,
            165,
            0,
        )

        for annotation in record.get(
            "annotations",
            [],
        ):
            box = [
                int(
                    round(
                        annotation[
                            "xmin"
                        ]
                    )
                ),
                int(
                    round(
                        annotation[
                            "ymin"
                        ]
                    )
                ),
                int(
                    round(
                        annotation[
                            "xmax"
                        ]
                    )
                ),
                int(
                    round(
                        annotation[
                            "ymax"
                        ]
                    )
                ),
            ]

            draw.rectangle(
                box,
                outline=color,
                width=line_width,
            )

            label = (
                f"{annotation['class_name']} "
                f"{annotation['confidence']:.2f}"
            )

            text_position = (
                box[0],
                max(
                    0,
                    box[1] - 22,
                ),
            )

            text_box = draw.textbbox(
                text_position,
                label,
            )

            draw.rectangle(
                text_box,
                fill=color,
            )

            draw.text(
                text_position,
                label,
                fill=(
                    0,
                    0,
                    0,
                ),
            )

        original_image_path = (
            Path(
                record[
                    "image_path"
                ]
            )
            .as_posix()
        )

        preview_suffix = (
            Path(
                original_image_path
            ).suffix
            or ".jpg"
        )

        preview_hash = (
            hashlib.sha256(
                original_image_path.encode(
                    "utf-8"
                )
            ).hexdigest()[:20]
        )

        preview_name = (
            preview_hash
            + preview_suffix.lower()
        )

        preview_path = (
            self.context.preview_directory
            / preview_name
        )

        preview.save(
            preview_path
        )

        return preview_path

    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================

    def save_predictions(
        self,
        records: list[dict[str, Any]],
    ) -> Path:
        predicted_boxes = sum(
            len(
                record.get(
                    "annotations",
                    [],
                )
            )
            for record
            in records
        )

        images_without_detections = sum(
            1
            for record
            in records
            if not record.get(
                "annotations"
            )
        )

        failed_images = sum(
            1
            for record
            in records
            if record.get(
                "status"
            )
            in {
                "prediction_error",
                "no_result",
            }
        )

        payload = {
            "source": (
                "dynamic_cumulative_"
                "fresh_batch_inference"
            ),
            "selected_folder": (
                self.context.selected_folder
            ),
            "mapped_label": (
                self.context.mapped_label
            ),
            "model_path": str(
                self.model_path
            ),
            "dataset_root": str(
                self.dataset_root
            ),
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "iou_threshold": (
                self.iou_threshold
            ),
            "image_size": (
                self.image_size
            ),
            "device": (
                self.device
            ),
            "summary": {
                "processed_images": len(
                    records
                ),
                "predicted_boxes": (
                    predicted_boxes
                ),
                "images_without_detections": (
                    images_without_detections
                ),
                "failed_images": (
                    failed_images
                ),
            },
            "images": (
                records
            ),
        }

        self.context.predictions_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            safe_write_json(
                destination=(
                    self.context.predictions_file
                ),
                payload=payload,
                retries=30,
            )

        except SafeIOError as exc:
            raise InferenceError(
                "Could not safely save "
                "predictions JSON: "
                f"{exc}"
            ) from exc

        return (
            self.context.predictions_file
        )

    # =====================================================
    # COMPLETE RUN
    # =====================================================

    def run(
        self,
        batch_size: int = 50,
        create_previews: bool = True,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise InferenceError(
                "batch_size must be greater than zero."
            )

        exclusion_files = (
            self.discover_exclusion_files()
        )

        discovery = (
            self.discover_fresh_images(
                exclusion_files
            )
        )

        fresh_images = discovery[
            "fresh_images"
        ]

        selected_images = fresh_images[
            :batch_size
        ]

        if not selected_images:
            raise InferenceError(
                "No fresh unprocessed images "
                "were found."
            )

        print(
            "Exclusion files selected:"
        )

        for path in exclusion_files:
            print(
                f"  - {path}"
            )

        print(
            "Excluded/manual/processed "
            "images skipped: "
            f"{len(discovery['skipped_images'])}"
        )

        print(
            "SHA-256 duplicate images skipped: "
            f"{len(discovery['duplicate_images'])}"
        )

        print(
            "Image hash errors: "
            f"{len(discovery['hash_errors'])}"
        )

        print(
            "Fresh unique images available: "
            f"{len(fresh_images)}"
        )

        print(
            "Fresh unique images selected: "
            f"{len(selected_images)}"
        )

        archive_directory = (
            self.archive_previous_batch()
        )

        if archive_directory is not None:
            print(
                "Previous working batch "
                "archived at: "
                f"{archive_directory}"
            )

        self.clear_working_batch()

        records = self.predict_images(
            selected_images
        )

        predictions_file = (
            self.save_predictions(
                records
            )
        )

        preview_count = 0

        if create_previews:
            for record in records:
                if (
                    record.get(
                        "status"
                    )
                    != "predicted"
                ):
                    continue

                try:
                    self.create_preview(
                        record
                    )

                    preview_count += 1

                except Exception as exc:
                    print(
                        "Preview failed for "
                        f"{record.get('image_name')}: "
                        f"{exc}"
                    )

        summary = {
            "status": "completed",
            "source": (
                "dynamic_cumulative_"
                "fresh_batch_inference"
            ),
            "selected_folder": (
                self.context.selected_folder
            ),
            "mapped_label": (
                self.context.mapped_label
            ),
            "model": str(
                self.model_path
            ),
            "excluded_image_paths": (
                discovery[
                    "excluded_path_count"
                ]
            ),
            "processed_images_skipped": len(
                discovery[
                    "skipped_images"
                ]
            ),
            "sha256_duplicates_skipped": len(
                discovery[
                    "duplicate_images"
                ]
            ),
            "processed_image_hashes": (
                discovery[
                    "processed_hash_count"
                ]
            ),
            "image_hash_errors": len(
                discovery[
                    "hash_errors"
                ]
            ),
            "fresh_unique_images_available": len(
                fresh_images
            ),
            "fresh_images_available": len(
                fresh_images
            ),
            "images_processed": len(
                records
            ),
            "predictions_json": str(
                predictions_file
            ),
            "preview_images_created": (
                preview_count
            ),
            "previous_batch_archive": (
                str(
                    archive_directory
                )
                if archive_directory
                else None
            ),
            "duplicate_images_skipped": (
                discovery[
                    "duplicate_images"
                ]
            ),
            "hash_errors": (
                discovery[
                    "hash_errors"
                ]
            ),
            "warning": (
                "Predictions are unverified. "
                "Review them before training use."
            ),
        }

        try:
            safe_write_json(
                destination=(
                    self.context.inference_summary_file
                ),
                payload=summary,
                retries=30,
            )

        except SafeIOError as exc:
            raise InferenceError(
                "Could not safely save "
                "inference summary: "
                f"{exc}"
            ) from exc

        return summary


# =========================================================
# DYNAMIC CONTEXT
# =========================================================

def build_context(
    selected_folder: str,
    dataset_root: str | Path,
) -> CategoryContext:
    project_root = Path(
        __file__
    ).resolve().parent.parent

    mapping_file = (
        project_root
        / "config"
        / "label_mapping.json"
    )

    mapping = load_label_mapping(
        mapping_file
    )

    mapped_label = get_mapped_label(
        selected_folder,
        mapping,
    )

    return CategoryContext.create(
        project_root=(
            project_root
        ),
        dataset_root=(
            dataset_root
        ),
        selected_folder=(
            selected_folder
        ),
        mapped_label=(
            mapped_label
        ),
    )


# =========================================================
# COMMAND-LINE ENTRY POINT
# =========================================================

def main() -> None:
    """
    Examples:

        python -m src.inference --folder S26_2 --dataset-root C:/path/to/dataset

        python -m src.inference --folder S25 --dataset-root C:/path/to/dataset

        python -m src.inference --folder S16/S16_2 --dataset-root C:/path/to/dataset

        python -m src.inference --folder S25 --dataset-root C:/path/to/dataset --batch-size 50
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run dynamic cumulative YOLO inference "
            "for one selected folder."
        )
    )

    parser.add_argument(
        "--folder",
        required=True,
        help=(
            "Dataset-relative folder, for example "
            "S26_2, S25 or S16/S16_2"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Absolute or relative path to the dataset root.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Fresh-image batch size. Default: 50",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Prediction confidence threshold. Default: 0.25",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS IoU threshold. Default: 0.70",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="YOLO inference image size. Default: 640",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device. Default: cpu",
    )

    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Do not create annotated preview images.",
    )

    args = parser.parse_args()

    try:
        context = build_context(
            args.folder,
            args.dataset_root,
        )

    except CategoryContextError as exc:
        raise InferenceError(
            str(
                exc
            )
        ) from exc

    context.ensure_working_directories()

    runner = DynamicYOLOInference(
        context=context,
        confidence_threshold=(
            args.confidence
        ),
        iou_threshold=(
            args.iou
        ),
        image_size=(
            args.image_size
        ),
        device=(
            args.device
        ),
    )

    result = runner.run(
        batch_size=(
            args.batch_size
        ),
        create_previews=(
            not args.no_previews
        ),
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()