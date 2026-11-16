from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.category_context import (
    CategoryContext,
    CategoryContextError,
    get_mapped_label,
    load_label_mapping,
)

from src.safe_io import (
    SafeIOError,
    safe_write_json,
)


class AnnotationMergeError(Exception):
    """Raised when annotation datasets cannot be merged safely."""

    pass


class AnnotationDatasetMerger:
    """
    Merge manual annotations and human-reviewed AI annotations
    into one cumulative verified training dataset.

    Supports any selected folder, including nested folders.

    Examples:
        S26_2
        S25
        S16/S16_2
        S22/s22_4

    Rules:
    - Manual Make Sense annotations are included.
    - Verified AI predictions are included.
    - Manual corrections are included.
    - Pending reviews are excluded.
    - Rejected reviews are excluded.
    - No-object records do not create boxes.
    - Unverified boxes are excluded.
    - Exact duplicate annotations are removed.
    """

    INCLUDED_REVIEW_STATUSES = {
        "verified",
        "corrected",
    }

    def __init__(
        self,
        manual_annotations_file: str | Path,
        review_files: list[str | Path],
        output_file: str | Path,
        selected_folder: str,
        mapped_label: str,
    ):
        self.manual_annotations_file = Path(
            manual_annotations_file
        ).resolve()

        self.output_file = Path(
            output_file
        ).resolve()

        self.selected_folder = str(
            selected_folder
        ).strip()

        self.mapped_label = str(
            mapped_label
        ).strip()

        self.review_files = (
            self._deduplicate_paths(
                review_files
            )
        )

        self._validate_inputs()

    # =====================================================
    # PATH HELPERS
    # =====================================================

    @staticmethod
    def _deduplicate_paths(
        paths: list[str | Path],
    ) -> list[Path]:
        """
        Remove duplicate file paths while preserving order.
        """

        unique_paths = []
        seen_paths = set()

        for value in paths:
            path = Path(
                value
            ).resolve()

            key = str(
                path
            ).casefold()

            if key in seen_paths:
                continue

            seen_paths.add(
                key
            )

            unique_paths.append(
                path
            )

        return unique_paths

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    def _validate_inputs(
        self,
    ) -> None:
        if not self.selected_folder:
            raise AnnotationMergeError(
                "Selected folder is empty."
            )

        if not self.mapped_label:
            raise AnnotationMergeError(
                "Mapped label is empty."
            )

        if not self.manual_annotations_file.is_file():
            raise AnnotationMergeError(
                "Manual annotation file not found: "
                f"{self.manual_annotations_file}"
            )

        existing_review_files = [
            path
            for path in self.review_files
            if path.is_file()
        ]

        self.review_files = (
            existing_review_files
        )

    # =====================================================
    # JSON
    # =====================================================

    @staticmethod
    def _load_json(
        file_path: Path,
    ) -> dict[str, Any]:
        try:
            data = json.loads(
                file_path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError as exc:
            raise AnnotationMergeError(
                "Invalid JSON file: "
                f"{file_path}"
            ) from exc

        except OSError as exc:
            raise AnnotationMergeError(
                "Could not read JSON file: "
                f"{file_path}"
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise AnnotationMergeError(
                "JSON root must be an object: "
                f"{file_path}"
            )

        images = data.get(
            "images"
        )

        if not isinstance(
            images,
            list,
        ):
            raise AnnotationMergeError(
                "JSON must contain an 'images' list: "
                f"{file_path}"
            )

        return data

    # =====================================================
    # IMAGE PATH
    # =====================================================

    @staticmethod
    def _normalize_image_path(
        image_path: Any,
    ) -> str:
        text = str(
            image_path
        ).strip()

        if not text:
            return ""

        return (
            Path(text)
            .as_posix()
        )

    # =====================================================
    # ANNOTATION KEY
    # =====================================================

    @staticmethod
    def _annotation_key(
        annotation: dict[str, Any],
    ) -> tuple:
        """
        Stable identity for exact duplicate detection.
        """

        return (
            str(
                annotation.get(
                    "class_name",
                    "",
                )
            )
            .strip()
            .casefold(),

            round(
                float(
                    annotation["xmin"]
                ),
                6,
            ),

            round(
                float(
                    annotation["ymin"]
                ),
                6,
            ),

            round(
                float(
                    annotation["xmax"]
                ),
                6,
            ),

            round(
                float(
                    annotation["ymax"]
                ),
                6,
            ),
        )

    # =====================================================
    # ANNOTATION VALIDATION
    # =====================================================

    def _validate_annotation(
        self,
        annotation: dict[str, Any],
        image_width: int,
        image_height: int,
        image_path: str,
    ) -> dict[str, Any]:
        class_name = str(
            annotation.get(
                "class_name",
                "",
            )
        ).strip()

        if not class_name:
            raise AnnotationMergeError(
                "Empty class_name for image: "
                f"{image_path}"
            )

        if class_name != self.mapped_label:
            raise AnnotationMergeError(
                "Unexpected class for selected folder "
                f"{self.selected_folder}: "
                f"{class_name}. "
                f"Expected: {self.mapped_label}"
            )

        try:
            xmin = float(
                annotation["xmin"]
            )

            ymin = float(
                annotation["ymin"]
            )

            xmax = float(
                annotation["xmax"]
            )

            ymax = float(
                annotation["ymax"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise AnnotationMergeError(
                "Invalid bounding box for image: "
                f"{image_path}"
            ) from exc

        if xmin < 0:
            raise AnnotationMergeError(
                "xmin is negative: "
                f"{image_path}"
            )

        if ymin < 0:
            raise AnnotationMergeError(
                "ymin is negative: "
                f"{image_path}"
            )

        if xmax <= xmin:
            raise AnnotationMergeError(
                "xmax must be greater than xmin: "
                f"{image_path}"
            )

        if ymax <= ymin:
            raise AnnotationMergeError(
                "ymax must be greater than ymin: "
                f"{image_path}"
            )

        if xmax > image_width:
            raise AnnotationMergeError(
                "Bounding box exceeds image width: "
                f"{image_path}"
            )

        if ymax > image_height:
            raise AnnotationMergeError(
                "Bounding box exceeds image height: "
                f"{image_path}"
            )

        normalized = deepcopy(
            annotation
        )

        normalized[
            "class_name"
        ] = class_name

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

        return normalized

    # =====================================================
    # NORMALIZE IMAGE RECORD
    # =====================================================

    def _normalize_image_record(
        self,
        raw_record: dict[str, Any],
        source_file: Path,
        source_type: str,
        require_verified: bool,
    ) -> dict[str, Any] | None:
        image_path = (
            self._normalize_image_path(
                raw_record.get(
                    "image_path",
                    "",
                )
            )
        )

        if not image_path:
            raise AnnotationMergeError(
                "image_path is empty in: "
                f"{source_file}"
            )

        try:
            image_width = int(
                raw_record[
                    "image_width"
                ]
            )

            image_height = int(
                raw_record[
                    "image_height"
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise AnnotationMergeError(
                "Invalid dimensions for image: "
                f"{image_path}"
            ) from exc

        if (
            image_width <= 0
            or image_height <= 0
        ):
            raise AnnotationMergeError(
                "Invalid dimensions for image: "
                f"{image_path}"
            )

        raw_annotations = raw_record.get(
            "annotations",
            [],
        )

        if not isinstance(
            raw_annotations,
            list,
        ):
            raise AnnotationMergeError(
                "annotations must be a list: "
                f"{image_path}"
            )

        annotations = []

        for raw_annotation in raw_annotations:
            if (
                require_verified
                and not raw_annotation.get(
                    "verified",
                    False,
                )
            ):
                continue

            annotation = (
                self._validate_annotation(
                    annotation=(
                        raw_annotation
                    ),
                    image_width=(
                        image_width
                    ),
                    image_height=(
                        image_height
                    ),
                    image_path=(
                        image_path
                    ),
                )
            )

            annotation[
                "training_source"
            ] = source_type

            annotation[
                "source_json"
            ] = str(
                source_file
            )

            annotations.append(
                annotation
            )

        if not annotations:
            return None

        return {
            "image_path": (
                image_path
            ),
            "image_name": (
                raw_record.get(
                    "image_name"
                )
                or Path(
                    image_path
                ).name
            ),
            "image_width": (
                image_width
            ),
            "image_height": (
                image_height
            ),
            "annotations": (
                annotations
            ),
            "source_json_files": [
                str(
                    source_file
                )
            ],
        }

    # =====================================================
    # MANUAL RECORDS
    # =====================================================

    def _load_manual_records(
        self,
    ) -> list[dict[str, Any]]:
        data = self._load_json(
            self.manual_annotations_file
        )

        records = []

        for raw_record in data[
            "images"
        ]:
            record = (
                self._normalize_image_record(
                    raw_record=(
                        raw_record
                    ),
                    source_file=(
                        self.manual_annotations_file
                    ),
                    source_type=(
                        "manual_make_sense"
                    ),
                    require_verified=False,
                )
            )

            if record is not None:
                records.append(
                    record
                )

        return records

    # =====================================================
    # REVIEW RECORDS
    # =====================================================

    def _load_review_records(
        self,
        review_file: Path,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, int],
    ]:
        """
        Supports:
        - review_store.json
        - verified_annotations.json
        """

        data = self._load_json(
            review_file
        )

        records = []

        statistics = {
            "total_records": 0,
            "included_records": 0,
            "pending_excluded": 0,
            "rejected_excluded": 0,
            "no_object_excluded": 0,
            "empty_excluded": 0,
        }

        is_review_store = (
            review_file.name.casefold()
            == "review_store.json"
        )

        for raw_record in data[
            "images"
        ]:
            statistics[
                "total_records"
            ] += 1

            if is_review_store:
                review_status = str(
                    raw_record.get(
                        "review_status",
                        "",
                    )
                ).strip().casefold()

                if (
                    review_status
                    == "pending"
                ):
                    statistics[
                        "pending_excluded"
                    ] += 1
                    continue

                if (
                    review_status
                    == "rejected"
                ):
                    statistics[
                        "rejected_excluded"
                    ] += 1
                    continue

                if (
                    review_status
                    == "no_object"
                ):
                    statistics[
                        "no_object_excluded"
                    ] += 1
                    continue

                if (
                    review_status
                    not in self.INCLUDED_REVIEW_STATUSES
                ):
                    statistics[
                        "empty_excluded"
                    ] += 1
                    continue

            record = (
                self._normalize_image_record(
                    raw_record=(
                        raw_record
                    ),
                    source_file=(
                        review_file
                    ),
                    source_type=(
                        "human_reviewed_ai"
                    ),
                    require_verified=True,
                )
            )

            if record is None:
                statistics[
                    "empty_excluded"
                ] += 1
                continue

            records.append(
                record
            )

            statistics[
                "included_records"
            ] += 1

        return (
            records,
            statistics,
        )

    # =====================================================
    # MERGE RECORD
    # =====================================================

    def _merge_record(
        self,
        combined_images: dict[
            str,
            dict[str, Any],
        ],
        record: dict[str, Any],
    ) -> int:
        """
        Merge one image record.

        Returns number of duplicate boxes removed.
        """

        image_path = (
            record[
                "image_path"
            ]
        )

        lookup_key = (
            image_path.casefold()
        )

        if (
            lookup_key
            not in combined_images
        ):
            combined_images[
                lookup_key
            ] = deepcopy(
                record
            )

            return 0

        existing = combined_images[
            lookup_key
        ]

        if (
            existing[
                "image_width"
            ]
            != record[
                "image_width"
            ]
            or existing[
                "image_height"
            ]
            != record[
                "image_height"
            ]
        ):
            raise AnnotationMergeError(
                "Dimension conflict for image: "
                f"{image_path}"
            )

        for source_file in record.get(
            "source_json_files",
            [],
        ):
            if (
                source_file
                not in existing[
                    "source_json_files"
                ]
            ):
                existing[
                    "source_json_files"
                ].append(
                    source_file
                )

        existing_keys = {
            self._annotation_key(
                annotation
            )
            for annotation
            in existing[
                "annotations"
            ]
        }

        duplicates_removed = 0

        for annotation in record[
            "annotations"
        ]:
            key = (
                self._annotation_key(
                    annotation
                )
            )

            if key in existing_keys:
                duplicates_removed += 1
                continue

            existing[
                "annotations"
            ].append(
                annotation
            )

            existing_keys.add(
                key
            )

        return duplicates_removed

    # =====================================================
    # MERGE ALL SOURCES
    # =====================================================

    def merge(
        self,
    ) -> dict[str, Any]:
        combined_images = {}

        duplicates_removed = 0

        manual_records = (
            self._load_manual_records()
        )

        manual_annotation_count = sum(
            len(
                record[
                    "annotations"
                ]
            )
            for record
            in manual_records
        )

        for record in manual_records:
            duplicates_removed += (
                self._merge_record(
                    combined_images,
                    record,
                )
            )

        review_file_summaries = []

        review_records_seen = 0
        review_records_included = 0
        review_annotations_included = 0

        pending_excluded = 0
        rejected_excluded = 0
        no_object_excluded = 0
        empty_excluded = 0

        for review_file in (
            self.review_files
        ):
            (
                review_records,
                statistics,
            ) = self._load_review_records(
                review_file
            )

            annotation_count = sum(
                len(
                    record[
                        "annotations"
                    ]
                )
                for record
                in review_records
            )

            review_records_seen += (
                statistics[
                    "total_records"
                ]
            )

            review_records_included += (
                statistics[
                    "included_records"
                ]
            )

            review_annotations_included += (
                annotation_count
            )

            pending_excluded += (
                statistics[
                    "pending_excluded"
                ]
            )

            rejected_excluded += (
                statistics[
                    "rejected_excluded"
                ]
            )

            no_object_excluded += (
                statistics[
                    "no_object_excluded"
                ]
            )

            empty_excluded += (
                statistics[
                    "empty_excluded"
                ]
            )

            file_duplicates = 0

            for record in review_records:
                file_duplicates += (
                    self._merge_record(
                        combined_images,
                        record,
                    )
                )

            duplicates_removed += (
                file_duplicates
            )

            review_file_summaries.append(
                {
                    "file": str(
                        review_file
                    ),
                    "records_seen": (
                        statistics[
                            "total_records"
                        ]
                    ),
                    "records_included": (
                        statistics[
                            "included_records"
                        ]
                    ),
                    "annotations_included": (
                        annotation_count
                    ),
                    "duplicates_removed": (
                        file_duplicates
                    ),
                }
            )

        output_images = sorted(
            combined_images.values(),
            key=lambda record: (
                record[
                    "image_path"
                ].casefold()
            ),
        )

        combined_annotations = sum(
            len(
                record[
                    "annotations"
                ]
            )
            for record
            in output_images
        )

        return {
            "source": (
                "dynamic_cumulative_verified_training_annotations"
            ),
            "selected_folder": (
                self.selected_folder
            ),
            "mapped_label": (
                self.mapped_label
            ),
            "manual_annotation_file": str(
                self.manual_annotations_file
            ),
            "review_files": [
                str(path)
                for path
                in self.review_files
            ],
            "review_file_summaries": (
                review_file_summaries
            ),
            "summary": {
                "manual_images": len(
                    manual_records
                ),
                "manual_annotations": (
                    manual_annotation_count
                ),
                "review_files_processed": len(
                    self.review_files
                ),
                "review_records_seen": (
                    review_records_seen
                ),
                "review_records_included": (
                    review_records_included
                ),
                "review_annotations_included": (
                    review_annotations_included
                ),
                "pending_records_excluded": (
                    pending_excluded
                ),
                "rejected_records_excluded": (
                    rejected_excluded
                ),
                "no_object_records_excluded": (
                    no_object_excluded
                ),
                "empty_records_excluded": (
                    empty_excluded
                ),
                "duplicate_annotations_removed": (
                    duplicates_removed
                ),
                "combined_images": len(
                    output_images
                ),
                "combined_annotations": (
                    combined_annotations
                ),
            },
            "images": (
                output_images
            ),
        }

    # =====================================================
    # SAVE
    # =====================================================

    def save(
        self,
        result: dict[str, Any],
    ) -> Path:
        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            return safe_write_json(
                destination=(
                    self.output_file
                ),
                payload=(
                    result
                ),
                retries=30,
            )

        except SafeIOError as exc:
            raise AnnotationMergeError(
                "Could not safely save "
                "combined annotations: "
                f"{self.output_file}: "
                f"{exc}"
            ) from exc

    def run(
        self,
    ) -> dict[str, Any]:
        result = self.merge()

        saved_path = self.save(
            result
        )

        return {
            "status": "completed",
            "selected_folder": (
                self.selected_folder
            ),
            "mapped_label": (
                self.mapped_label
            ),
            "output_file": str(
                saved_path
            ),
            **result[
                "summary"
            ],
        }


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
# BACKWARD-COMPATIBLE PATHS
# =========================================================

def get_manual_annotation_file(
    context: CategoryContext,
) -> Path:
    """
    Preserve existing S26_2 manual annotation file.
    """

    if (
        context.selected_folder
        == "S26_2"
    ):
        legacy_file = (
            context.project_root
            / "data"
            / "normalized_annotations"
            / "s26_2_manual_annotations.json"
        )

        if legacy_file.is_file():
            return legacy_file

    return (
        context
        .normalized_annotations_file
    )


def get_combined_output_file(
    context: CategoryContext,
) -> Path:
    """
    Preserve existing S26_2 combined annotation file.
    """

    if (
        context.selected_folder
        == "S26_2"
    ):
        legacy_file = (
            context.project_root
            / "data"
            / "combined_annotations"
            / "s26_2_training_annotations.json"
        )

        if (
            legacy_file.is_file()
            or legacy_file.parent.exists()
        ):
            return legacy_file

    return (
        context
        .combined_annotations_file
    )


# =========================================================
# REVIEW SOURCE DISCOVERY
# =========================================================

def discover_review_files(
    context: CategoryContext,
) -> list[Path]:
    """
    Discover review sources for ONLY selected folder.

    Includes:
    - archived review_store.json
    - current review_store.json
    - current verified_annotations.json
    """

    discovered = []

    archive_root = (
        context
        .prediction_archive_root
    )

    if archive_root.is_dir():
        discovered.extend(
            sorted(
                archive_root.rglob(
                    "review_store.json"
                )
            )
        )

    current_review_store = (
        context
        .review_store_file
    )

    if current_review_store.is_file():
        discovered.append(
            current_review_store
        )

    current_verified_file = (
        context
        .verified_annotations_file
    )

    if current_verified_file.is_file():
        discovered.append(
            current_verified_file
        )

    # -----------------------------------------------------
    # S26_2 legacy locations are already identical for
    # review/prediction data, so no special copy/move is
    # required.
    # -----------------------------------------------------

    unique_files = []

    seen_paths = set()

    for path in discovered:
        resolved = (
            path.resolve()
        )

        key = str(
            resolved
        ).casefold()

        if key in seen_paths:
            continue

        seen_paths.add(
            key
        )

        unique_files.append(
            resolved
        )

    return unique_files


# =========================================================
# CLI
# =========================================================

def main():
    """
    Examples:

        python -m src.annotation_merger --folder S26_2

        python -m src.annotation_merger --folder S25

        python -m src.annotation_merger --folder S16/S16_2
    """

    parser = argparse.ArgumentParser(
        description=(
            "Merge cumulative verified "
            "annotations for selected folder."
        )
    )

    parser.add_argument(
        "--folder",
        required=True,
        help=(
            "Dataset-relative folder such as "
            "S26_2, S25 or S16/S16_2"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Absolute or relative path to the dataset root.",
    )

    args = parser.parse_args()

    try:
        context = build_context(
            args.folder,
            args.dataset_root,
        )

    except CategoryContextError as exc:
        raise AnnotationMergeError(
            str(
                exc
            )
        ) from exc

    context.ensure_working_directories()

    manual_file = (
        get_manual_annotation_file(
            context
        )
    )

    output_file = (
        get_combined_output_file(
            context
        )
    )

    review_files = (
        discover_review_files(
            context
        )
    )

    print(
        "Selected folder: "
        f"{context.selected_folder}"
    )

    print(
        "Mapped label: "
        f"{context.mapped_label}"
    )

    print(
        "Manual annotation file:"
    )

    print(
        f"  - {manual_file}"
    )

    print(
        "Review files selected:"
    )

    if review_files:
        for review_file in (
            review_files
        ):
            print(
                f"  - {review_file}"
            )

    else:
        print(
            "  - No review files found."
        )

    merger = AnnotationDatasetMerger(
        manual_annotations_file=(
            manual_file
        ),
        review_files=(
            review_files
        ),
        output_file=(
            output_file
        ),
        selected_folder=(
            context.selected_folder
        ),
        mapped_label=(
            context.mapped_label
        ),
    )

    result = merger.run()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()