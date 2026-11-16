from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.safe_io import (
    SafeIOError,
    safe_write_json,
)


class AnnotationReviewError(Exception):
    """Raised when prediction review data cannot be handled safely."""

    pass


class AnnotationReviewManager:
    """
    Review AI predictions for one selected folder.

    The manager is path-driven, so every selected
    folder/subfolder can have its own:

        predictions.json
        review_store.json
        verified_annotations.json

    Existing app.py calls remain compatible.
    """

    VALID_REVIEW_STATUSES = {
        "pending",
        "verified",
        "corrected",
        "rejected",
        "no_object",
    }

    def __init__(
        self,
        dataset_root: str | Path,
        predictions_json: str | Path,
        output_file: str | Path,
        valid_labels: list[str] | set[str],
    ):
        self.dataset_root = Path(
            dataset_root
        ).resolve()

        self.predictions_json = Path(
            predictions_json
        ).resolve()

        self.output_file = Path(
            output_file
        ).resolve()

        self.valid_labels = {
            str(label).strip()
            for label in valid_labels
            if str(label).strip()
        }

        self._validate_configuration()

        self.prediction_payload = (
            self._load_predictions()
        )

        self.prediction_lookup = {
            self._normalize_image_path(
                record.get(
                    "image_path",
                    "",
                )
            ): record
            for record
            in self.prediction_payload[
                "images"
            ]
            if (
                isinstance(
                    record,
                    dict,
                )
                and self._normalize_image_path(
                    record.get(
                        "image_path",
                        "",
                    )
                )
            )
        }

        self.review_payload = (
            self._load_or_initialize_review_store()
        )

        self._sync_with_predictions()

    # =====================================================
    # CONFIGURATION
    # =====================================================

    def _validate_configuration(
        self,
    ) -> None:
        if not self.dataset_root.is_dir():
            raise AnnotationReviewError(
                "Dataset root not found: "
                f"{self.dataset_root}"
            )

        if not self.predictions_json.is_file():
            raise AnnotationReviewError(
                "Predictions JSON not found: "
                f"{self.predictions_json}"
            )

        if not self.valid_labels:
            raise AnnotationReviewError(
                "At least one valid label "
                "is required."
            )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =====================================================
    # IMAGE PATH
    # =====================================================

    @staticmethod
    def _normalize_image_path(
        value: Any,
    ) -> str:
        text = str(
            value
        ).strip()

        if not text:
            return ""

        return (
            Path(
                text
            )
            .as_posix()
        )

    # =====================================================
    # JSON
    # =====================================================

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:
        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError as exc:
            raise AnnotationReviewError(
                "Invalid JSON: "
                f"{path}"
            ) from exc

        except OSError as exc:
            raise AnnotationReviewError(
                "Could not read JSON: "
                f"{path}"
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise AnnotationReviewError(
                "JSON root must be "
                "an object: "
                f"{path}"
            )

        return data

    # =====================================================
    # PREDICTIONS
    # =====================================================

    def _load_predictions(
        self,
    ) -> dict[str, Any]:
        data = self._load_json(
            self.predictions_json
        )

        images = data.get(
            "images"
        )

        if not isinstance(
            images,
            list,
        ):
            raise AnnotationReviewError(
                "Predictions JSON must contain "
                "an 'images' list."
            )

        return data

    # =====================================================
    # REVIEW STORE
    # =====================================================

    def _load_or_initialize_review_store(
        self,
    ) -> dict[str, Any]:
        if self.output_file.is_file():
            data = self._load_json(
                self.output_file
            )

            if not isinstance(
                data.get(
                    "images"
                ),
                list,
            ):
                raise AnnotationReviewError(
                    "Review store must contain "
                    "an 'images' list."
                )

            return data

        return {
            "source": (
                "human_review_of_ai_predictions"
            ),
            "predictions_json": str(
                self.predictions_json
            ),
            "valid_labels": sorted(
                self.valid_labels
            ),
            "images": [],
        }

    # =====================================================
    # CREATE REVIEW RECORD
    # =====================================================

    @staticmethod
    def _prediction_record_to_review(
        record: dict[str, Any],
    ) -> dict[str, Any]:
        result = deepcopy(
            record
        )

        result[
            "review_status"
        ] = "pending"

        result[
            "review_note"
        ] = None

        result[
            "original_annotations"
        ] = deepcopy(
            record.get(
                "annotations",
                [],
            )
        )

        return result

    # =====================================================
    # SYNC REVIEW STORE WITH CURRENT PREDICTIONS
    # =====================================================

    def _sync_with_predictions(
        self,
    ) -> None:
        existing = {
            self._normalize_image_path(
                record.get(
                    "image_path",
                    "",
                )
            ): record
            for record
            in self.review_payload.get(
                "images",
                [],
            )
            if (
                isinstance(
                    record,
                    dict,
                )
                and self._normalize_image_path(
                    record.get(
                        "image_path",
                        "",
                    )
                )
            )
        }

        synchronized = []

        for prediction in (
            self.prediction_payload[
                "images"
            ]
        ):
            image_path = (
                self._normalize_image_path(
                    prediction.get(
                        "image_path",
                        "",
                    )
                )
            )

            if not image_path:
                continue

            if image_path in existing:
                synchronized.append(
                    existing[
                        image_path
                    ]
                )

            else:
                synchronized.append(
                    self._prediction_record_to_review(
                        prediction
                    )
                )

        self.review_payload[
            "predictions_json"
        ] = str(
            self.predictions_json
        )

        self.review_payload[
            "valid_labels"
        ] = sorted(
            self.valid_labels
        )

        self.review_payload[
            "images"
        ] = synchronized

        self._save_review_store()

    # =====================================================
    # SAVE REVIEW STORE
    # =====================================================

    def _save_review_store(
        self,
    ) -> Path:
        """
        Persist review state using central
        Windows-safe JSON handling.
        """

        try:
            return safe_write_json(
                destination=(
                    self.output_file
                ),
                payload=(
                    self.review_payload
                ),
                retries=30,
            )

        except SafeIOError as exc:
            raise AnnotationReviewError(
                "Could not safely save "
                "review store: "
                f"{self.output_file}: "
                f"{exc}"
            ) from exc


    def _find_review_record(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        normalized = (
            self._normalize_image_path(
                image_path
            )
        )

        for record in (
            self.review_payload[
                "images"
            ]
        ):
            record_path = (
                self._normalize_image_path(
                    record.get(
                        "image_path",
                        "",
                    )
                )
            )

            if record_path == normalized:
                return record

        raise AnnotationReviewError(
            "Image not found in review queue: "
            f"{normalized}"
        )

    # =====================================================
    # VALIDATE ANNOTATIONS
    # =====================================================

    def _validate_annotations(
        self,
        image_path: str,
        annotations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        record = self._find_review_record(
            image_path
        )

        image_width = int(
            record.get(
                "image_width",
                0,
            )
        )

        image_height = int(
            record.get(
                "image_height",
                0,
            )
        )

        if (
            image_width <= 0
            or image_height <= 0
        ):
            source_image = (
                self.dataset_root
                / Path(
                    image_path
                )
            )

            if not source_image.is_file():
                raise AnnotationReviewError(
                    "Source image not found: "
                    f"{source_image}"
                )

            from PIL import Image

            with Image.open(
                source_image
            ) as source:
                (
                    image_width,
                    image_height,
                ) = source.size

            record[
                "image_width"
            ] = image_width

            record[
                "image_height"
            ] = image_height

        validated = []

        for annotation in annotations:
            class_name = str(
                annotation.get(
                    "class_name",
                    "",
                )
            ).strip()

            if (
                class_name
                not in self.valid_labels
            ):
                raise AnnotationReviewError(
                    "Invalid class "
                    f"'{class_name}'. "
                    "Expected: "
                    + ", ".join(
                        sorted(
                            self.valid_labels
                        )
                    )
                )

            try:
                xmin = float(
                    annotation[
                        "xmin"
                    ]
                )

                ymin = float(
                    annotation[
                        "ymin"
                    ]
                )

                xmax = float(
                    annotation[
                        "xmax"
                    ]
                )

                ymax = float(
                    annotation[
                        "ymax"
                    ]
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise AnnotationReviewError(
                    "Invalid bounding box for "
                    f"{image_path}"
                ) from exc

            if xmin < 0:
                raise AnnotationReviewError(
                    "xmin cannot be negative: "
                    f"{image_path}"
                )

            if ymin < 0:
                raise AnnotationReviewError(
                    "ymin cannot be negative: "
                    f"{image_path}"
                )

            if xmax <= xmin:
                raise AnnotationReviewError(
                    "xmax must be greater "
                    "than xmin: "
                    f"{image_path}"
                )

            if ymax <= ymin:
                raise AnnotationReviewError(
                    "ymax must be greater "
                    "than ymin: "
                    f"{image_path}"
                )

            if xmax > image_width:
                raise AnnotationReviewError(
                    "Bounding box exceeds "
                    "image width: "
                    f"{image_path}"
                )

            if ymax > image_height:
                raise AnnotationReviewError(
                    "Bounding box exceeds "
                    "image height: "
                    f"{image_path}"
                )

            normalized = deepcopy(
                annotation
            )

            normalized[
                "class_name"
            ] = class_name

            normalized[
                "class_id"
            ] = annotation.get(
                "class_id"
            )

            normalized[
                "confidence"
            ] = annotation.get(
                "confidence"
            )

            normalized[
                "xmin"
            ] = xmin

            normalized[
                "ymin"
            ] = ymin

            normalized[
                "xmax"
            ] = xmax

            normalized[
                "ymax"
            ] = ymax

            normalized[
                "verified"
            ] = True

            validated.append(
                normalized
            )

        return validated

    # =====================================================
    # SUMMARY
    # =====================================================

    def get_summary(
        self,
    ) -> dict[str, int]:
        counts = {
            status: 0
            for status
            in self.VALID_REVIEW_STATUSES
        }

        for record in (
            self.review_payload[
                "images"
            ]
        ):
            status = str(
                record.get(
                    "review_status",
                    "pending",
                )
            ).strip().casefold()

            if status not in counts:
                status = "pending"

            counts[
                status
            ] += 1

        return {
            "total_images": len(
                self.review_payload[
                    "images"
                ]
            ),
            **counts,
        }

    # =====================================================
    # REVIEW QUEUE
    # =====================================================

    def get_review_queue(
        self,
        status: str | None = None,
        lowest_confidence_first: bool = True,
    ) -> list[dict[str, Any]]:
        if status is not None:
            status = (
                str(
                    status
                )
                .strip()
                .casefold()
            )

            if (
                status
                not in self.VALID_REVIEW_STATUSES
            ):
                raise AnnotationReviewError(
                    "Invalid review status: "
                    f"{status}"
                )

        records = [
            record
            for record
            in self.review_payload[
                "images"
            ]
            if (
                status is None
                or str(
                    record.get(
                        "review_status",
                        "pending",
                    )
                )
                .strip()
                .casefold()
                == status
            )
        ]

        if lowest_confidence_first:

            def confidence_key(
                record: dict[str, Any],
            ) -> tuple[int, float, str]:
                confidence = record.get(
                    "highest_confidence"
                )

                image_path = (
                    self._normalize_image_path(
                        record.get(
                            "image_path",
                            "",
                        )
                    )
                )

                if confidence is None:
                    return (
                        0,
                        0.0,
                        image_path,
                    )

                try:
                    return (
                        1,
                        float(
                            confidence
                        ),
                        image_path,
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    return (
                        0,
                        0.0,
                        image_path,
                    )

            records.sort(
                key=confidence_key
            )

        return records

    # =====================================================
    # VERIFY AI PREDICTION
    # =====================================================

    def verify_prediction(
        self,
        image_path: str | Path,
        note: str | None = None,
    ) -> Path:
        record = self._find_review_record(
            image_path
        )

        annotations = record.get(
            "annotations",
            [],
        )

        if (
            not isinstance(
                annotations,
                list,
            )
            or not annotations
        ):
            raise AnnotationReviewError(
                "Cannot verify a prediction "
                "with no bounding boxes. "
                "Use correction or no-object instead."
            )

        verified = (
            self._validate_annotations(
                record[
                    "image_path"
                ],
                annotations,
            )
        )

        for annotation in verified:
            annotation[
                "source"
            ] = (
                annotation.get(
                    "source"
                )
                or "ai_prediction"
            )

            annotation[
                "verified"
            ] = True

        record[
            "annotations"
        ] = verified

        record[
            "review_status"
        ] = "verified"

        record[
            "review_note"
        ] = note

        return self._save_review_store()

    # =====================================================
    # REJECT
    # =====================================================

    def reject_prediction(
        self,
        image_path: str | Path,
        note: str | None = None,
    ) -> Path:
        record = self._find_review_record(
            image_path
        )

        record[
            "review_status"
        ] = "rejected"

        record[
            "review_note"
        ] = note

        record[
            "annotations"
        ] = []

        return self._save_review_store()

    # =====================================================
    # NO OBJECT
    # =====================================================

    def mark_no_object(
        self,
        image_path: str | Path,
        note: str | None = None,
    ) -> Path:
        record = self._find_review_record(
            image_path
        )

        record[
            "review_status"
        ] = "no_object"

        record[
            "review_note"
        ] = note

        record[
            "annotations"
        ] = []

        return self._save_review_store()

    # =====================================================
    # MANUAL CORRECTION
    # =====================================================

    def save_corrected_annotations(
        self,
        image_path: str | Path,
        annotations: list[dict[str, Any]],
        note: str | None = None,
    ) -> Path:
        normalized_path = (
            self._normalize_image_path(
                image_path
            )
        )

        if not annotations:
            raise AnnotationReviewError(
                "At least one corrected bounding "
                "box is required. "
                "Use Mark No Object when appropriate."
            )

        validated = (
            self._validate_annotations(
                normalized_path,
                annotations,
            )
        )

        for annotation in validated:
            annotation[
                "source"
            ] = "manual_correction"

            annotation[
                "verified"
            ] = True

            annotation[
                "class_id"
            ] = annotation.get(
                "class_id"
            )

            annotation[
                "confidence"
            ] = None

        record = self._find_review_record(
            normalized_path
        )

        record[
            "annotations"
        ] = validated

        record[
            "review_status"
        ] = "corrected"

        record[
            "review_note"
        ] = note

        record[
            "highest_confidence"
        ] = None

        record[
            "prediction_count"
        ] = len(
            validated
        )

        return self._save_review_store()

    # =====================================================
    # EXPORT VERIFIED ANNOTATIONS
    # =====================================================

    def export_verified_annotations(
        self,
        output_file: str | Path,
    ) -> Path:
        destination = Path(
            output_file
        ).resolve()

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        exported_images = []

        annotation_count = 0

        for record in (
            self.review_payload[
                "images"
            ]
        ):
            status = str(
                record.get(
                    "review_status",
                    "pending",
                )
            ).strip().casefold()

            if status not in {
                "verified",
                "corrected",
            }:
                continue

            annotations = [
                deepcopy(
                    annotation
                )
                for annotation
                in record.get(
                    "annotations",
                    [],
                )
                if annotation.get(
                    "verified",
                    False,
                )
            ]

            if not annotations:
                continue

            image_path = (
                self._normalize_image_path(
                    record.get(
                        "image_path",
                        "",
                    )
                )
            )

            exported_images.append(
                {
                    "image_path": (
                        image_path
                    ),
                    "image_name": (
                        record.get(
                            "image_name"
                        )
                        or Path(
                            image_path
                        ).name
                    ),
                    "image_width": int(
                        record[
                            "image_width"
                        ]
                    ),
                    "image_height": int(
                        record[
                            "image_height"
                        ]
                    ),
                    "annotations": (
                        annotations
                    ),
                }
            )

            annotation_count += len(
                annotations
            )

        payload = {
            "source": (
                "human_reviewed_ai_predictions"
            ),
            "images": (
                exported_images
            ),
            "summary": {
                "verified_images": len(
                    exported_images
                ),
                "verified_annotations": (
                    annotation_count
                ),
            },
        }

        try:
            return safe_write_json(
                destination=(
                    destination
                ),
                payload=(
                    payload
                ),
                retries=30,
            )

        except SafeIOError as exc:
            raise AnnotationReviewError(
                "Could not safely export "
                "verified annotations: "
                f"{destination}: "
                f"{exc}"
            ) from exc

