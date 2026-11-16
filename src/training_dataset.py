from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.category_context import (
    CategoryContext,
    CategoryContextError,
    get_mapped_label,
    load_label_mapping,
)

from src.safe_io import (
    SafeIOError,
    safe_write_json,
    safe_write_text,
)


class TrainingDatasetError(Exception):
    """Raised when a YOLO dataset cannot be generated safely."""

    pass


class YOLOTrainingDatasetBuilder:
    """
    Build an isolated YOLO dataset for one
    selected folder or subfolder.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        output_root: str | Path,
        class_names: list[str],
        selected_folder: str,
        validation_ratio: float = 0.20,
        random_seed: int = 42,
    ):
        self.dataset_root = Path(
            dataset_root
        ).resolve()

        self.output_root = Path(
            output_root
        ).resolve()

        self.class_names = [
            str(name).strip()
            for name in class_names
            if str(name).strip()
        ]

        self.selected_folder = (
            str(selected_folder)
            .strip()
            .replace(
                "\\",
                "/",
            )
            .strip("/")
        )

        self.validation_ratio = float(
            validation_ratio
        )

        self.random_seed = int(
            random_seed
        )

        self.class_to_id = {
            name: index
            for index, name
            in enumerate(
                self.class_names
            )
        }

        self._validate_configuration()

    # =====================================================
    # CONFIGURATION
    # =====================================================

    def _validate_configuration(
        self,
    ) -> None:
        if not self.dataset_root.is_dir():
            raise TrainingDatasetError(
                "Dataset root not found: "
                f"{self.dataset_root}"
            )

        if not self.selected_folder:
            raise TrainingDatasetError(
                "Selected folder is empty."
            )

        selected_path = (
            self.dataset_root
            / Path(
                self.selected_folder
            )
        ).resolve()

        if not selected_path.is_dir():
            raise TrainingDatasetError(
                "Selected dataset folder not found: "
                f"{selected_path}"
            )

        if not self.class_names:
            raise TrainingDatasetError(
                "At least one class name is required."
            )

        if (
            len(
                set(
                    self.class_names
                )
            )
            != len(
                self.class_names
            )
        ):
            raise TrainingDatasetError(
                "Duplicate class names are not allowed."
            )

        if not (
            0.0
            < self.validation_ratio
            < 1.0
        ):
            raise TrainingDatasetError(
                "validation_ratio must be "
                "between 0 and 1."
            )

    # =====================================================
    # LOAD ANNOTATIONS
    # =====================================================

    @staticmethod
    def load_annotations(
        annotation_json: str | Path,
    ) -> dict[str, Any]:
        annotation_path = Path(
            annotation_json
        ).resolve()

        if not annotation_path.is_file():
            raise TrainingDatasetError(
                "Training annotation JSON "
                "not found: "
                f"{annotation_path}"
            )

        try:
            data = json.loads(
                annotation_path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError as exc:
            raise TrainingDatasetError(
                "Invalid annotation JSON: "
                f"{annotation_path}"
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise TrainingDatasetError(
                "Annotation JSON root "
                "must be an object."
            )

        images = data.get(
            "images"
        )

        if not isinstance(
            images,
            list,
        ):
            raise TrainingDatasetError(
                "Annotation JSON must contain "
                "an 'images' list."
            )

        if not images:
            raise TrainingDatasetError(
                "No annotated images were found."
            )

        return data

    # =====================================================
    # BOUNDING-BOX CONVERSION
    # =====================================================

    @staticmethod
    def convert_bbox_to_yolo(
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        image_width: int,
        image_height: int,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Convert pixel-space XYXY coordinates into normalized
        YOLO coordinates:

            x_center
            y_center
            width
            height
        """

        if (
            image_width <= 0
            or image_height <= 0
        ):
            raise TrainingDatasetError(
                "Image dimensions must be "
                "greater than zero."
            )

        if (
            xmin < 0
            or ymin < 0
            or xmax <= xmin
            or ymax <= ymin
        ):
            raise TrainingDatasetError(
                "Invalid bounding-box coordinates."
            )

        if (
            xmax > image_width
            or ymax > image_height
        ):
            raise TrainingDatasetError(
                "Bounding box exceeds "
                "image boundaries."
            )

        box_width = (
            xmax
            - xmin
        )

        box_height = (
            ymax
            - ymin
        )

        x_center = (
            xmin
            + box_width / 2.0
        )

        y_center = (
            ymin
            + box_height / 2.0
        )

        return (
            x_center / image_width,
            y_center / image_height,
            box_width / image_width,
            box_height / image_height,
        )

    # =====================================================
    # RECORD VALIDATION
    # =====================================================

    def _validate_image_record(
        self,
        record: dict[str, Any],
    ) -> None:
        required_fields = {
            "image_path",
            "image_width",
            "image_height",
            "annotations",
        }

        missing_fields = (
            required_fields
            - set(
                record
            )
        )

        if missing_fields:
            raise TrainingDatasetError(
                "Image record missing fields: "
                + ", ".join(
                    sorted(
                        missing_fields
                    )
                )
            )

        image_path = (
            Path(
                str(
                    record[
                        "image_path"
                    ]
                )
            )
            .as_posix()
        )

        selected_prefix = (
            self.selected_folder
            .casefold()
            + "/"
        )

        if not (
            image_path
            .casefold()
            .startswith(
                selected_prefix
            )
        ):
            raise TrainingDatasetError(
                "Image does not belong to "
                "selected folder "
                f"{self.selected_folder}: "
                f"{image_path}"
            )

        try:
            image_width = int(
                record[
                    "image_width"
                ]
            )

            image_height = int(
                record[
                    "image_height"
                ]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TrainingDatasetError(
                "Invalid image dimensions: "
                f"{image_path}"
            ) from exc

        if (
            image_width <= 0
            or image_height <= 0
        ):
            raise TrainingDatasetError(
                "Invalid image dimensions: "
                f"{image_path}"
            )

        annotations = record[
            "annotations"
        ]

        if not isinstance(
            annotations,
            list,
        ):
            raise TrainingDatasetError(
                "annotations must be a list: "
                f"{image_path}"
            )

        if not annotations:
            raise TrainingDatasetError(
                "No bounding boxes found: "
                f"{image_path}"
            )

    # =====================================================
    # YOLO LABEL LINES
    # =====================================================

    def _build_yolo_lines(
        self,
        record: dict[str, Any],
    ) -> list[str]:
        self._validate_image_record(
            record
        )

        image_width = int(
            record[
                "image_width"
            ]
        )

        image_height = int(
            record[
                "image_height"
            ]
        )

        lines = []

        for annotation in record[
            "annotations"
        ]:
            class_name = str(
                annotation.get(
                    "class_name",
                    "",
                )
            ).strip()

            if class_name not in self.class_to_id:
                raise TrainingDatasetError(
                    "Unknown class "
                    f"'{class_name}' in "
                    f"{record['image_path']}"
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
                raise TrainingDatasetError(
                    "Invalid bounding-box data in "
                    f"{record['image_path']}"
                ) from exc

            (
                x_center,
                y_center,
                normalized_width,
                normalized_height,
            ) = self.convert_bbox_to_yolo(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                image_width=image_width,
                image_height=image_height,
            )

            class_id = (
                self.class_to_id[
                    class_name
                ]
            )

            lines.append(
                (
                    f"{class_id} "
                    f"{x_center:.8f} "
                    f"{y_center:.8f} "
                    f"{normalized_width:.8f} "
                    f"{normalized_height:.8f}"
                )
            )

        return lines

    # =====================================================
    # SHA-256 IMAGE HASH
    # =====================================================

    @staticmethod
    def calculate_sha256(
        image_path: Path,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Calculate exact SHA-256 content hash.

        Source images are read only.
        Original images are never deleted or modified.
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
            raise TrainingDatasetError(
                "Could not calculate SHA-256 for "
                f"{image_path}: {exc}"
            ) from exc

        return digest.hexdigest()


    # =====================================================
    # DEDUPLICATE ANNOTATED RECORDS
    # =====================================================

    def deduplicate_records(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """
        Keep only the first annotated image record for
        each unique SHA-256 image hash.

        Deduplication happens before train/validation split.

        This prevents the same physical image content from
        appearing in both training and validation sets under
        different filenames or folder paths.

        Returns:

            unique_records
            duplicate_records
        """

        unique_records = []

        duplicate_records = []

        hash_to_retained_path = {}

        for record in records:
            self._validate_image_record(
                record
            )

            relative_path = Path(
                str(
                    record[
                        "image_path"
                    ]
                )
            )

            source_image = (
                self.dataset_root
                / relative_path
            )

            if not source_image.is_file():
                raise TrainingDatasetError(
                    "Source image not found during "
                    "SHA-256 deduplication: "
                    f"{source_image}"
                )

            image_hash = (
                self.calculate_sha256(
                    source_image
                )
            )

            normalized_record = dict(
                record
            )

            normalized_record[
                "sha256"
            ] = image_hash

            if (
                image_hash
                in hash_to_retained_path
            ):
                duplicate_records.append(
                    {
                        "image_path": (
                            relative_path.as_posix()
                        ),
                        "sha256": (
                            image_hash
                        ),
                        "duplicate_of": (
                            hash_to_retained_path[
                                image_hash
                            ]
                        ),
                        "bounding_boxes": len(
                            record.get(
                                "annotations",
                                [],
                            )
                        ),
                    }
                )

                continue

            hash_to_retained_path[
                image_hash
            ] = relative_path.as_posix()

            unique_records.append(
                normalized_record
            )

        if len(unique_records) < 2:
            raise TrainingDatasetError(
                "At least 2 unique annotated images "
                "are required after SHA-256 "
                "deduplication."
            )

        return (
            unique_records,
            duplicate_records,
        )


    # =====================================================
    # TRAIN / VALIDATION SPLIT
    # =====================================================
    def split_records(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        if len(records) < 2:
            raise TrainingDatasetError(
                "At least 2 annotated images "
                "are required."
            )

        shuffled_records = list(
            records
        )

        random_generator = random.Random(
            self.random_seed
        )

        random_generator.shuffle(
            shuffled_records
        )

        validation_count = round(
            len(shuffled_records)
            * self.validation_ratio
        )

        validation_count = max(
            1,
            validation_count,
        )

        validation_count = min(
            validation_count,
            len(shuffled_records) - 1,
        )

        validation_records = shuffled_records[
            :validation_count
        ]

        training_records = shuffled_records[
            validation_count:
        ]

        return (
            training_records,
            validation_records,
        )
    # =====================================================
    # OUTPUT DIRECTORIES
    # =====================================================

    def _prepare_directories(
        self,
    ) -> dict[str, Path]:
        """
        Recreate only the generated training output for the
        selected folder.

        Original source images are not modified.
        """

        if self.output_root.exists():
            shutil.rmtree(
                self.output_root
            )

        directories = {
            "train_images": (
                self.output_root
                / "images"
                / "train"
            ),
            "val_images": (
                self.output_root
                / "images"
                / "val"
            ),
            "train_labels": (
                self.output_root
                / "labels"
                / "train"
            ),
            "val_labels": (
                self.output_root
                / "labels"
                / "val"
            ),
        }

        for directory in (
            directories.values()
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        return directories

    # =====================================================
    # COLLISION-SAFE GENERATED NAME
    # =====================================================

    @staticmethod
    def _generated_name(
        relative_path: Path,
    ) -> str:
        """
        Preserve folder identity inside generated filename.

        Example:

            S16/S16_2/image001.jpg

        becomes:

            S16__S16_2__image001.jpg
        """

        parent = "__".join(
            part.replace(
                " ",
                "_",
            )
            for part
            in relative_path.parts[
                :-1
            ]
        )

        if parent:
            return (
                f"{parent}__"
                f"{relative_path.name}"
            )

        return relative_path.name

    # =====================================================
    # WRITE ONE RECORD
    # =====================================================

    def _write_record(
        self,
        record: dict[str, Any],
        image_directory: Path,
        label_directory: Path,
    ) -> dict[str, Any]:
        relative_path = Path(
            str(
                record[
                    "image_path"
                ]
            )
        )

        source_image = (
            self.dataset_root
            / relative_path
        )

        if not source_image.is_file():
            raise TrainingDatasetError(
                "Source image not found: "
                f"{source_image}"
            )

        generated_name = (
            self._generated_name(
                relative_path
            )
        )

        destination_image = (
            image_directory
            / generated_name
        )

        destination_label = (
            label_directory
            / (
                Path(
                    generated_name
                ).stem
                + ".txt"
            )
        )

        lines = self._build_yolo_lines(
            record
        )

        shutil.copy2(
            source_image,
            destination_image,
        )

        try:
            safe_write_text(
                destination=(
                    destination_label
                ),
                content=(
                    "\n".join(
                        lines
                    )
                    + "\n"
                ),
                retries=30,
            )

        except SafeIOError as exc:
            raise TrainingDatasetError(
                "Could not safely write YOLO "
                "label file: "
                f"{destination_label}: "
                f"{exc}"
            ) from exc

        return {
            "source_image": (
                relative_path.as_posix()
            ),
            "generated_image": (
                destination_image.name
            ),
            "generated_label": (
                destination_label.name
            ),
            "bounding_boxes": len(
                record[
                    "annotations"
                ]
            ),
            "sha256": (
                record.get(
                    "sha256"
                )
            ),
        }

    # =====================================================
    # DATASET YAML
    # =====================================================

    def _write_dataset_yaml(
        self,
    ) -> Path:
        dataset_yaml = (
            self.output_root
            / "dataset.yaml"
        )

        data = {
            "path": str(
                self.output_root
            ),
            "train": (
                "images/train"
            ),
            "val": (
                "images/val"
            ),
            "names": {
                index: name
                for name, index
                in self.class_to_id.items()
            },
        }

        dataset_yaml.write_text(
            yaml.safe_dump(
                data,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        return dataset_yaml

    # =====================================================
    # CLASS MAP
    # =====================================================

    def _write_class_map(
        self,
    ) -> Path:
        class_map_file = (
            self.output_root
            / "class_names.json"
        )

        class_map = {
            str(index): name
            for name, index
            in self.class_to_id.items()
        }

        try:
            safe_write_json(
                destination=(
                    class_map_file
                ),
                payload=(
                    class_map
                ),
                retries=30,
            )

        except SafeIOError as exc:
            raise TrainingDatasetError(
                "Could not safely write "
                "class_names.json: "
                f"{exc}"
            ) from exc

        return class_map_file

    # =====================================================
    # COMPLETE BUILD
    # =====================================================

    def build_from_json(
        self,
        annotation_json: str | Path,
    ) -> dict[str, Any]:
        annotation_data = (
            self.load_annotations(
                annotation_json
            )
        )

        original_records = annotation_data[
            "images"
        ]

        (
            records,
            duplicate_records,
        ) = self.deduplicate_records(
            original_records
        )

        print(
            "Original annotated images: "
            f"{len(original_records)}"
        )

        print(
            "Unique annotated images retained: "
            f"{len(records)}"
        )

        print(
            "SHA-256 duplicate annotated images "
            "skipped: "
            f"{len(duplicate_records)}"
        )

        (
            training_records,
            validation_records,
        ) = self.split_records(
            records
        )

        directories = (
            self._prepare_directories()
        )

        training_manifest = []

        validation_manifest = []

        for record in training_records:
            training_manifest.append(
                self._write_record(
                    record=record,
                    image_directory=(
                        directories[
                            "train_images"
                        ]
                    ),
                    label_directory=(
                        directories[
                            "train_labels"
                        ]
                    ),
                )
            )

        for record in validation_records:
            validation_manifest.append(
                self._write_record(
                    record=record,
                    image_directory=(
                        directories[
                            "val_images"
                        ]
                    ),
                    label_directory=(
                        directories[
                            "val_labels"
                        ]
                    ),
                )
            )

        dataset_yaml = (
            self._write_dataset_yaml()
        )

        class_map = (
            self._write_class_map()
        )

        total_boxes = sum(
            len(
                record[
                    "annotations"
                ]
            )
            for record
            in records
        )

        manifest = {
            "source_type": (
                "dynamic_manual_plus_"
                "human_reviewed"
            ),
            "selected_folder": (
                self.selected_folder
            ),
            "source_annotation_json": str(
                Path(
                    annotation_json
                ).resolve()
            ),
            "dataset_root": str(
                self.dataset_root
            ),
            "output_root": str(
                self.output_root
            ),
            "random_seed": (
                self.random_seed
            ),
            "validation_ratio": (
                self.validation_ratio
            ),
            "classes": (
                self.class_to_id
            ),
            "original_annotated_images": len(
                original_records
            ),
            "unique_images_after_sha256_deduplication": len(
                records
            ),
            "sha256_duplicate_images_skipped": len(
                duplicate_records
            ),
            "duplicate_records": (
                duplicate_records
            ),
            "total_images": len(
                records
            ),
            "total_bounding_boxes": (
                total_boxes
            ),
            "training_images": len(
                training_manifest
            ),
            "validation_images": len(
                validation_manifest
            ),
            "training": (
                training_manifest
            ),
            "validation": (
                validation_manifest
            ),
        }

        manifest_path = (
            self.output_root
            / "dataset_manifest.json"
        )

        try:
            safe_write_json(
                destination=(
                    manifest_path
                ),
                payload=(
                    manifest
                ),
                retries=30,
            )

        except SafeIOError as exc:
            raise TrainingDatasetError(
                "Could not safely write "
                "dataset manifest: "
                f"{exc}"
            ) from exc

        return {
            "status": "completed",
            "selected_folder": (
                self.selected_folder
            ),
            "annotation_source": str(
                Path(
                    annotation_json
                ).resolve()
            ),
            "output_root": str(
                self.output_root
            ),
            "dataset_yaml": str(
                dataset_yaml
            ),
            "class_map": str(
                class_map
            ),
            "manifest": str(
                manifest_path
            ),
            "total_images": len(
                records
            ),
            "total_bounding_boxes": (
                total_boxes
            ),
            "training_images": len(
                training_manifest
            ),
            "validation_images": len(
                validation_manifest
            ),
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
# BACKWARD-COMPATIBLE INPUT
# =========================================================

def get_combined_annotation_file(
    context: CategoryContext,
) -> Path:
    """
    Preserve the existing S26_2 combined JSON location.

    New folders use dynamic CategoryContext locations.
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

        if legacy_file.is_file():
            return legacy_file

    return (
        context
        .combined_annotations_file
    )


# =========================================================
# CLI
# =========================================================

def main() -> None:
    """
    Examples:

        python -m src.training_dataset --folder S26_2 --dataset-root C:/path/to/dataset

        python -m src.training_dataset --folder S25 --dataset-root C:/path/to/dataset

        python -m src.training_dataset --folder S16/S16_2 --dataset-root C:/path/to/dataset
    """

    parser = argparse.ArgumentParser(
        description=(
            "Build a YOLO dataset for "
            "a selected folder/subfolder."
        )
    )

    parser.add_argument(
        "--folder",
        required=True,
        help=(
            "Dataset-relative folder, "
            "for example S26_2, S25 "
            "or S16/S16_2"
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
        raise TrainingDatasetError(
            str(
                exc
            )
        ) from exc

    context.ensure_working_directories()

    annotation_json = (
        get_combined_annotation_file(
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
        "Combined annotation JSON:"
    )

    print(
        f"  - {annotation_json}"
    )

    print(
        "Training output:"
    )

    print(
        f"  - {context.training_root}"
    )

    builder = (
        YOLOTrainingDatasetBuilder(
            dataset_root=(
                context.dataset_root
            ),
            output_root=(
                context.training_root
            ),
            class_names=[
                context.mapped_label
            ],
            selected_folder=(
                context.selected_folder
            ),
            validation_ratio=0.20,
            random_seed=42,
        )
    )

    result = builder.build_from_json(
        annotation_json
    )

    result[
        "mapped_label"
    ] = context.mapped_label

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()