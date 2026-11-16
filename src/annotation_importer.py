from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from src.category_context import (
    CategoryContext,
    CategoryContextError,
    get_mapped_label,
    load_label_mapping,
    normalize_relative_folder,
)

from src.safe_io import (
    SafeIOError,
    safe_write_json,
)


class AnnotationImportError(Exception):
    """Raised when annotation import or validation fails."""

    pass


class MakeSenseCSVImporter:
    """
    Import and validate one or more Make Sense AI CSV exports.

    Supports:
    - Dynamic folders/subfolders
    - Multiple CSV files
    - Exact annotation deduplication
    - EXIF-oriented image dimensions
    """

    REQUIRED_COLUMNS = {
        "label_name",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "image_name",
        "image_width",
        "image_height",
    }

    SUPPORTED_IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".jfif",
        ".png",
        ".bmp",
        ".webp",
    }

    IGNORED_DIRECTORY_NAMES = {
        "image_zip",
        "output",
        "models",
        "data",
        ".venv",
        "__pycache__",
    }

    def __init__(
        self,
        dataset_root: str | Path,
        valid_labels: list[str] | set[str],
        selected_folder: str | Path | None = None,
    ):
        self.dataset_root = Path(
            dataset_root
        ).resolve()

        self.valid_labels = {
            str(label).strip()
            for label in valid_labels
            if str(label).strip()
        }

        self.selected_folder = None

        if selected_folder is not None:
            self.selected_folder = (
                normalize_relative_folder(
                    selected_folder
                )
            )

        if not self.dataset_root.is_dir():
            raise AnnotationImportError(
                "Dataset root not found: "
                f"{self.dataset_root}"
            )

        if not self.valid_labels:
            raise AnnotationImportError(
                "Valid label list is empty."
            )

        self.image_index = (
            self._build_image_index()
        )

    # =====================================================
    # IMAGE SEARCH ROOT
    # =====================================================

    @property
    def image_search_root(
        self,
    ) -> Path:
        if self.selected_folder:
            path = (
                self.dataset_root
                / Path(
                    self.selected_folder
                )
            )

            if not path.is_dir():
                raise AnnotationImportError(
                    "Selected dataset folder "
                    "not found: "
                    f"{path}"
                )

            return path

        return self.dataset_root

    # =====================================================
    # GENERATED DIRECTORY EXCLUSION
    # =====================================================

    def _is_ignored_path(
        self,
        image_path: Path,
    ) -> bool:
        try:
            relative_path = (
                image_path.relative_to(
                    self.dataset_root
                )
            )

        except ValueError:
            return True

        ignored = {
            name.casefold()
            for name
            in self.IGNORED_DIRECTORY_NAMES
        }

        return any(
            part.casefold()
            in ignored
            for part
            in relative_path.parts[:-1]
        )

    # =====================================================
    # IMAGE INDEX
    # =====================================================

    def _build_image_index(
        self,
    ) -> dict[str, list[Path]]:
        image_index: dict[
            str,
            list[Path],
        ] = defaultdict(
            list
        )

        for image_path in (
            self.image_search_root.rglob("*")
        ):
            if not image_path.is_file():
                continue

            if (
                image_path.suffix.casefold()
                not in self.SUPPORTED_IMAGE_EXTENSIONS
            ):
                continue

            if self._is_ignored_path(
                image_path
            ):
                continue

            image_index[
                image_path.name.casefold()
            ].append(
                image_path
            )

        return dict(
            image_index
        )

    # =====================================================
    # CSV DELIMITER
    # =====================================================

    @staticmethod
    def _detect_delimiter(
        csv_path: Path,
    ) -> str:
        sample = csv_path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )[:8192]

        if not sample.strip():
            raise AnnotationImportError(
                "CSV is empty: "
                f"{csv_path.name}"
            )

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=",\t;",
            )

            return dialect.delimiter

        except csv.Error:
            return ","

    # =====================================================
    # NUMERIC VALUE
    # =====================================================

    @staticmethod
    def _to_number(
        value: Any,
        field_name: str,
        row_number: int,
        source_file: str,
    ) -> float:
        text = str(
            value
        ).strip()

        if not text:
            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                f"{field_name} is empty."
            )

        try:
            return float(
                text
            )

        except ValueError as exc:
            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                f"{field_name} is not numeric: "
                f"{text}"
            ) from exc

    # =====================================================
    # IMAGE MATCHING
    # =====================================================

    def _match_image(
        self,
        image_name: str,
        row_number: int,
        source_file: str,
    ) -> Path:
        filename = Path(
            str(
                image_name
            ).strip()
        ).name

        if not filename:
            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                "image_name is empty."
            )

        matches = self.image_index.get(
            filename.casefold(),
            [],
        )

        if not matches:
            folder_info = (
                self.selected_folder
                or "dataset root"
            )

            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                f"image not found in "
                f"{folder_info}: "
                f"{filename}"
            )

        if len(matches) > 1:
            paths = ", ".join(
                path.relative_to(
                    self.dataset_root
                ).as_posix()
                for path
                in matches
            )

            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                "ambiguous image filename "
                f"{filename}. "
                f"Matches: {paths}"
            )

        return matches[0]

    # =====================================================
    # BOUNDING BOX VALIDATION
    # =====================================================

    @staticmethod
    def _validate_bbox(
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        image_width: int,
        image_height: int,
        row_number: int,
        source_file: str,
    ) -> None:
        prefix = (
            f"{source_file}, "
            f"row {row_number}: "
        )

        if (
            image_width <= 0
            or image_height <= 0
        ):
            raise AnnotationImportError(
                prefix
                + "invalid image dimensions."
            )

        if (
            xmin < 0
            or ymin < 0
        ):
            raise AnnotationImportError(
                prefix
                + "bounding-box minimum "
                "cannot be negative."
            )

        if (
            xmax <= xmin
            or ymax <= ymin
        ):
            raise AnnotationImportError(
                prefix
                + "invalid bounding-box size."
            )

        if (
            xmax > image_width
            or ymax > image_height
        ):
            raise AnnotationImportError(
                prefix
                + "bounding box exceeds "
                + "oriented image dimensions "
                + f"{image_width}x{image_height}."
            )

    # =====================================================
    # PARSE ONE ROW
    # =====================================================

    def _parse_row(
        self,
        row: dict[str, Any],
        row_number: int,
        source_file: str,
    ) -> dict[str, Any]:
        cleaned_row = {
            str(key).strip(): value
            for key, value
            in row.items()
            if key is not None
        }

        label_name = str(
            cleaned_row.get(
                "label_name",
                "",
            )
        ).strip()

        image_name = str(
            cleaned_row.get(
                "image_name",
                "",
            )
        ).strip()

        if not label_name:
            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                "label_name is empty."
            )

        if (
            label_name
            not in self.valid_labels
        ):
            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                f"invalid label '{label_name}'. "
                "Expected one of: "
                + ", ".join(
                    sorted(
                        self.valid_labels
                    )
                )
            )

        image_path = self._match_image(
            image_name,
            row_number,
            source_file,
        )

        # -------------------------------------------------
        # CSV DIMENSIONS
        # -------------------------------------------------

        csv_width_value = self._to_number(
            cleaned_row.get(
                "image_width"
            ),
            "image_width",
            row_number,
            source_file,
        )

        csv_height_value = self._to_number(
            cleaned_row.get(
                "image_height"
            ),
            "image_height",
            row_number,
            source_file,
        )

        if (
            not csv_width_value.is_integer()
            or not csv_height_value.is_integer()
        ):
            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                "image dimensions must "
                "be integers."
            )

        csv_width = int(
            csv_width_value
        )

        csv_height = int(
            csv_height_value
        )

        # -------------------------------------------------
        # EXIF-AWARE ACTUAL DIMENSIONS
        # -------------------------------------------------

        try:
            with Image.open(
                image_path
            ) as image:
                (
                    stored_width,
                    stored_height,
                ) = image.size

                oriented_image = (
                    ImageOps.exif_transpose(
                        image
                    )
                )

                (
                    actual_width,
                    actual_height,
                ) = oriented_image.size

        except Exception as exc:
            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                "could not open source image "
                f"{image_path.name}: "
                f"{exc}"
            ) from exc

        if (
            csv_width != actual_width
            or csv_height != actual_height
        ):
            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                "image dimensions mismatch. "
                f"CSV={csv_width}x{csv_height}, "
                "EXIF-oriented actual="
                f"{actual_width}x{actual_height}, "
                "stored="
                f"{stored_width}x{stored_height}."
            )

        # -------------------------------------------------
        # BOUNDING BOX
        # -------------------------------------------------

        bbox_x = self._to_number(
            cleaned_row.get(
                "bbox_x"
            ),
            "bbox_x",
            row_number,
            source_file,
        )

        bbox_y = self._to_number(
            cleaned_row.get(
                "bbox_y"
            ),
            "bbox_y",
            row_number,
            source_file,
        )

        bbox_width = self._to_number(
            cleaned_row.get(
                "bbox_width"
            ),
            "bbox_width",
            row_number,
            source_file,
        )

        bbox_height = self._to_number(
            cleaned_row.get(
                "bbox_height"
            ),
            "bbox_height",
            row_number,
            source_file,
        )

        if (
            bbox_width <= 0
            or bbox_height <= 0
        ):
            raise AnnotationImportError(
                f"{source_file}, "
                f"row {row_number}: "
                "bbox width/height must "
                "be greater than zero."
            )

        xmin = bbox_x
        ymin = bbox_y

        xmax = (
            bbox_x
            + bbox_width
        )

        ymax = (
            bbox_y
            + bbox_height
        )

        self._validate_bbox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            image_width=actual_width,
            image_height=actual_height,
            row_number=row_number,
            source_file=source_file,
        )

        relative_path = (
            image_path
            .relative_to(
                self.dataset_root
            )
            .as_posix()
        )

        return {
            "image_name": (
                image_path.name
            ),
            "image_path": (
                relative_path
            ),
            "image_width": (
                actual_width
            ),
            "image_height": (
                actual_height
            ),
            "class_name": (
                label_name
            ),
            "xmin": (
                xmin
            ),
            "ymin": (
                ymin
            ),
            "xmax": (
                xmax
            ),
            "ymax": (
                ymax
            ),
            "source": "manual",
            "verified": True,
            "confidence": None,
            "source_file": (
                source_file
            ),
            "source_row": (
                row_number
            ),
        }

    # =====================================================
    # IMPORT ONE CSV
    # =====================================================

    def import_csv(
        self,
        csv_file: str | Path,
    ) -> dict[str, Any]:
        csv_path = Path(
            csv_file
        ).resolve()

        if not csv_path.is_file():
            raise AnnotationImportError(
                "Annotation CSV not found: "
                f"{csv_path}"
            )

        source_file = (
            csv_path.name
        )

        delimiter = (
            self._detect_delimiter(
                csv_path
            )
        )

        imported_rows = []
        errors = []

        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_handle:
            reader = csv.DictReader(
                csv_handle,
                delimiter=delimiter,
            )

            if reader.fieldnames is None:
                raise AnnotationImportError(
                    f"{source_file}: "
                    "CSV header is missing."
                )

            normalized_headers = [
                (
                    header.strip()
                    if header
                    else ""
                )
                for header
                in reader.fieldnames
            ]

            reader.fieldnames = (
                normalized_headers
            )

            missing_columns = (
                self.REQUIRED_COLUMNS
                - set(
                    normalized_headers
                )
            )

            if missing_columns:
                raise AnnotationImportError(
                    f"{source_file}: "
                    "missing required columns: "
                    + ", ".join(
                        sorted(
                            missing_columns
                        )
                    )
                )

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                if not any(
                    str(value).strip()
                    for value
                    in row.values()
                    if value is not None
                ):
                    continue

                try:
                    annotation = (
                        self._parse_row(
                            row,
                            row_number,
                            source_file,
                        )
                    )

                    imported_rows.append(
                        annotation
                    )

                except Exception as exc:
                    errors.append(
                        {
                            "source_file": (
                                source_file
                            ),
                            "row_number": (
                                row_number
                            ),
                            "error": str(
                                exc
                            ),
                        }
                    )

        return {
            "source_file": str(
                csv_path
            ),
            "source_name": (
                source_file
            ),
            "rows": (
                imported_rows
            ),
            "errors": (
                errors
            ),
            "successful_rows": len(
                imported_rows
            ),
            "error_count": len(
                errors
            ),
        }

    # =====================================================
    # ANNOTATION KEY
    # =====================================================

    @staticmethod
    def _annotation_key(
        annotation: dict[str, Any],
    ) -> tuple:
        return (
            str(
                annotation[
                    "image_path"
                ]
            ).casefold(),
            str(
                annotation[
                    "class_name"
                ]
            ).casefold(),
            round(
                float(
                    annotation[
                        "xmin"
                    ]
                ),
                6,
            ),
            round(
                float(
                    annotation[
                        "ymin"
                    ]
                ),
                6,
            ),
            round(
                float(
                    annotation[
                        "xmax"
                    ]
                ),
                6,
            ),
            round(
                float(
                    annotation[
                        "ymax"
                    ]
                ),
                6,
            ),
        )

    # =====================================================
    # IMPORT MULTIPLE CSV FILES
    # =====================================================

    def import_multiple_csv(
        self,
        csv_files: list[
            str | Path
        ],
    ) -> dict[str, Any]:
        if not csv_files:
            raise AnnotationImportError(
                "No CSV files supplied."
            )

        unique_csv_paths = []
        seen_csv_paths = set()

        for value in csv_files:
            csv_path = Path(
                value
            ).resolve()

            key = str(
                csv_path
            ).casefold()

            if key in seen_csv_paths:
                continue

            seen_csv_paths.add(
                key
            )

            unique_csv_paths.append(
                csv_path
            )

        combined_rows = []
        combined_errors = []
        source_summaries = []

        for csv_path in unique_csv_paths:
            try:
                result = self.import_csv(
                    csv_path
                )

                combined_rows.extend(
                    result[
                        "rows"
                    ]
                )

                combined_errors.extend(
                    result[
                        "errors"
                    ]
                )

                source_summaries.append(
                    {
                        "source_file": (
                            result[
                                "source_name"
                            ]
                        ),
                        "successful_rows": (
                            result[
                                "successful_rows"
                            ]
                        ),
                        "error_count": (
                            result[
                                "error_count"
                            ]
                        ),
                    }
                )

            except Exception as exc:
                combined_errors.append(
                    {
                        "source_file": (
                            csv_path.name
                        ),
                        "row_number": None,
                        "error": str(
                            exc
                        ),
                    }
                )

                source_summaries.append(
                    {
                        "source_file": (
                            csv_path.name
                        ),
                        "successful_rows": 0,
                        "error_count": 1,
                    }
                )

        deduplicated_rows = []
        seen_annotations = set()
        duplicates_removed = 0

        for annotation in combined_rows:
            key = self._annotation_key(
                annotation
            )

            if key in seen_annotations:
                duplicates_removed += 1
                continue

            seen_annotations.add(
                key
            )

            deduplicated_rows.append(
                annotation
            )

        images = self._group_by_image(
            deduplicated_rows
        )

        return {
            "format": (
                "make_sense_csv_multi"
            ),
            "selected_folder": (
                self.selected_folder
            ),
            "valid_labels": sorted(
                self.valid_labels
            ),
            "source_files": [
                str(path)
                for path
                in unique_csv_paths
            ],
            "source_file_count": len(
                unique_csv_paths
            ),
            "source_summaries": (
                source_summaries
            ),
            "raw_valid_annotations": len(
                combined_rows
            ),
            "duplicate_annotations_removed": (
                duplicates_removed
            ),
            "imported_annotations": len(
                deduplicated_rows
            ),
            "imported_images": len(
                images
            ),
            "error_count": len(
                combined_errors
            ),
            "errors": (
                combined_errors
            ),
            "images": (
                images
            ),
        }

    # =====================================================
    # GROUP BY IMAGE
    # =====================================================

    @staticmethod
    def _group_by_image(
        imported_rows: list[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:
        grouped = {}

        for annotation in imported_rows:
            image_path = (
                annotation[
                    "image_path"
                ]
            )

            if image_path not in grouped:
                grouped[
                    image_path
                ] = {
                    "image_path": (
                        image_path
                    ),
                    "image_name": (
                        annotation[
                            "image_name"
                        ]
                    ),
                    "image_width": (
                        annotation[
                            "image_width"
                        ]
                    ),
                    "image_height": (
                        annotation[
                            "image_height"
                        ]
                    ),
                    "annotations": [],
                    "source_files": [],
                }

            record = grouped[
                image_path
            ]

            source_file = annotation[
                "source_file"
            ]

            if (
                source_file
                not in record[
                    "source_files"
                ]
            ):
                record[
                    "source_files"
                ].append(
                    source_file
                )

            record[
                "annotations"
            ].append(
                {
                    "class_name": (
                        annotation[
                            "class_name"
                        ]
                    ),
                    "xmin": (
                        annotation[
                            "xmin"
                        ]
                    ),
                    "ymin": (
                        annotation[
                            "ymin"
                        ]
                    ),
                    "xmax": (
                        annotation[
                            "xmax"
                        ]
                    ),
                    "ymax": (
                        annotation[
                            "ymax"
                        ]
                    ),
                    "source": (
                        annotation[
                            "source"
                        ]
                    ),
                    "verified": True,
                    "confidence": None,
                    "source_file": (
                        source_file
                    ),
                    "source_row": (
                        annotation[
                            "source_row"
                        ]
                    ),
                }
            )

        return sorted(
            grouped.values(),
            key=lambda record: (
                record[
                    "image_path"
                ].casefold()
            ),
        )

    # =====================================================
    # SAVE NORMALIZED JSON
    # =====================================================

    @staticmethod
    def save_normalized_json(
        import_result: dict[str, Any],
        output_file: str | Path,
    ) -> Path:
        output_path = Path(
            output_file
        ).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            return safe_write_json(
                destination=(
                    output_path
                ),
                payload=(
                    import_result
                ),
                retries=30,
            )

        except SafeIOError as exc:
            raise AnnotationImportError(
                "Could not safely save "
                "normalized annotations: "
                f"{output_path}: "
                f"{exc}"
            ) from exc


# =========================================================
# DYNAMIC CONTEXT
# =========================================================

def build_context_from_cli(
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
# CLI
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import Make Sense annotations "
            "for a selected dataset folder."
        )
    )

    parser.add_argument(
        "--folder",
        required=True,
        help=(
            "Dataset-relative folder, for example "
            "S26_2 or S20_ALL/..."
        ),
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Absolute or relative path to the dataset root.",
    )

    args = parser.parse_args()

    try:
        context = build_context_from_cli(
            args.folder,
            args.dataset_root,
        )

    except CategoryContextError as exc:
        raise AnnotationImportError(
            str(
                exc
            )
        ) from exc

    context.ensure_working_directories()

    csv_files = sorted(
        context
        .manual_annotations_directory
        .rglob(
            "*.csv"
        )
    )

    if not csv_files:
        raise AnnotationImportError(
            "No CSV files found under: "
            f"{context.manual_annotations_directory}"
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
        "CSV files selected:"
    )

    for csv_path in csv_files:
        print(
            f"  - {csv_path.name}"
        )

    importer = MakeSenseCSVImporter(
        dataset_root=(
            context.dataset_root
        ),
        valid_labels=[
            context.mapped_label
        ],
        selected_folder=(
            context.selected_folder
        ),
    )

    result = (
        importer.import_multiple_csv(
            csv_files
        )
    )

    if (
        context.selected_folder
        == "S26_2"
    ):
        legacy_output = (
            context.project_root
            / "data"
            / "normalized_annotations"
            / "s26_2_manual_annotations.json"
        )

        output_file = (
            legacy_output
            if legacy_output.exists()
            else context.normalized_annotations_file
        )

    else:
        output_file = (
            context
            .normalized_annotations_file
        )

    saved_path = (
        importer.save_normalized_json(
            result,
            output_file,
        )
    )

    summary = {
        "status": "completed",
        "selected_folder": (
            context.selected_folder
        ),
        "mapped_label": (
            context.mapped_label
        ),
        "source_files": (
            result[
                "source_file_count"
            ]
        ),
        "imported_images": (
            result[
                "imported_images"
            ]
        ),
        "imported_annotations": (
            result[
                "imported_annotations"
            ]
        ),
        "duplicates_removed": (
            result[
                "duplicate_annotations_removed"
            ]
        ),
        "errors": (
            result[
                "error_count"
            ]
        ),
        "normalized_json": str(
            saved_path
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    if result[
        "error_count"
    ]:
        print(
            "\nFirst import errors:"
        )

        for error in result[
            "errors"
        ][:10]:
            print(
                error
            )


if __name__ == "__main__":
    main()