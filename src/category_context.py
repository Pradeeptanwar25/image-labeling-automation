from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CategoryContextError(Exception):
    """Raised when selected-folder context is invalid."""

    pass


# =========================================================
# PATH / NAME HELPERS
# =========================================================

def normalize_relative_folder(
    folder: str | Path,
) -> str:
    """
    Normalize selected folder to a safe relative POSIX path.

    Examples:
        S26_2
        S25/S25_1
        S25/Variant_A/Type_2
    """

    text = str(folder).strip()

    if not text:
        raise CategoryContextError(
            "Selected folder cannot be empty."
        )

    text = text.replace(
        "\\",
        "/",
    )

    path = Path(text)

    if path.is_absolute():
        raise CategoryContextError(
            "Selected folder must be relative "
            "to the dataset root."
        )

    if ".." in path.parts:
        raise CategoryContextError(
            "Selected folder cannot contain '..'."
        )

    cleaned_parts = [
        part.strip()
        for part in path.parts
        if part.strip()
        and part.strip() != "."
    ]

    if not cleaned_parts:
        raise CategoryContextError(
            "Selected folder is invalid."
        )

    return Path(
        *cleaned_parts
    ).as_posix()


def safe_key(
    folder: str | Path,
) -> str:
    """
    Convert selected folder into a filename-safe key.

    Examples:
        S26_2
            -> S26_2

        S25/S25_2
            -> S25__S25_2
    """

    relative_folder = (
        normalize_relative_folder(
            folder
        )
    )

    key = relative_folder.replace(
        "/",
        "__",
    )

    key = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        key,
    )

    key = key.strip(
        "._-"
    )

    if not key:
        raise CategoryContextError(
            "Could not generate category key."
        )

    return key


# =========================================================
# CATEGORY CONTEXT
# =========================================================

@dataclass(frozen=True)
class CategoryContext:
    """
    Central context for one selected image folder.

    Example selected_folder:
        S26_2

    or:
        S25/S25_2

    All training/review/model/output data for that
    selected folder stays isolated.
    """

    project_root: Path
    dataset_root: Path
    selected_folder: str
    mapped_label: str

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        dataset_root: str | Path,
        selected_folder: str | Path,
        mapped_label: str,
    ) -> "CategoryContext":

        project_root_path = Path(
            project_root
        ).resolve()

        dataset_root_path = Path(
            dataset_root
        ).resolve()

        relative_folder = (
            normalize_relative_folder(
                selected_folder
            )
        )

        label = str(
            mapped_label
        ).strip()

        if not project_root_path.is_dir():
            raise CategoryContextError(
                "Project root not found: "
                f"{project_root_path}"
            )

        if not dataset_root_path.is_dir():
            raise CategoryContextError(
                "Dataset root not found: "
                f"{dataset_root_path}"
            )

        selected_path = (
            dataset_root_path
            / Path(
                relative_folder
            )
        ).resolve()

        try:
            selected_path.relative_to(
                dataset_root_path
            )

        except ValueError as exc:
            raise CategoryContextError(
                "Selected folder is outside "
                "the dataset root."
            ) from exc

        if not selected_path.is_dir():
            raise CategoryContextError(
                "Selected dataset folder not found: "
                f"{selected_path}"
            )

        if not label:
            raise CategoryContextError(
                "Selected folder does not have "
                "a mapped label."
            )

        return cls(
            project_root=(
                project_root_path
            ),
            dataset_root=(
                dataset_root_path
            ),
            selected_folder=(
                relative_folder
            ),
            mapped_label=(
                label
            ),
        )

    # =====================================================
    # IDENTITY
    # =====================================================

    @property
    def folder_parts(
        self,
    ) -> tuple[str, ...]:

        return tuple(
            Path(
                self.selected_folder
            ).parts
        )

    @property
    def category_key(
        self,
    ) -> str:
        """
        Filename-safe identifier.
        """

        return safe_key(
            self.selected_folder
        )

    @property
    def display_name(
        self,
    ) -> str:
        """
        Friendly frontend display.

        Example:
            S25 / S25_2
        """

        return " / ".join(
            self.folder_parts
        )

    @property
    def selected_dataset_path(
        self,
    ) -> Path:

        return (
            self.dataset_root
            / Path(
                self.selected_folder
            )
        )

    # =====================================================
    # MANUAL ANNOTATIONS
    # =====================================================

    @property
    def manual_annotations_directory(
        self,
    ) -> Path:
        """
        Example:
            data/manual_annotations/S25/S25_2/
        """

        return (
            self.project_root
            / "data"
            / "manual_annotations"
            / Path(
                self.selected_folder
            )
        )

    # =====================================================
    # NORMALIZED ANNOTATIONS
    # =====================================================

    @property
    def normalized_annotations_directory(
        self,
    ) -> Path:

        return (
            self.project_root
            / "data"
            / "normalized_annotations"
            / Path(
                self.selected_folder
            )
        )

    @property
    def normalized_annotations_file(
        self,
    ) -> Path:

        return (
            self.normalized_annotations_directory
            / "manual_annotations.json"
        )

    # =====================================================
    # COMBINED ANNOTATIONS
    # =====================================================

    @property
    def combined_annotations_directory(
        self,
    ) -> Path:

        return (
            self.project_root
            / "data"
            / "combined_annotations"
            / Path(
                self.selected_folder
            )
        )

    @property
    def combined_annotations_file(
        self,
    ) -> Path:

        return (
            self.combined_annotations_directory
            / "training_annotations.json"
        )

    # =====================================================
    # TRAINING DATA
    # =====================================================

    @property
    def training_root(
        self,
    ) -> Path:

        return (
            self.project_root
            / "data"
            / "training"
            / Path(
                self.selected_folder
            )
        )

    @property
    def dataset_yaml(
        self,
    ) -> Path:

        return (
            self.training_root
            / "dataset.yaml"
        )

    @property
    def dataset_manifest(
        self,
    ) -> Path:

        return (
            self.training_root
            / "dataset_manifest.json"
        )

    @property
    def class_map_file(
        self,
    ) -> Path:

        return (
            self.training_root
            / "class_names.json"
        )

    # =====================================================
    # REVIEWS
    # =====================================================

    @property
    def review_root(
        self,
    ) -> Path:

        return (
            self.project_root
            / "data"
            / "reviews"
            / Path(
                self.selected_folder
            )
        )

    @property
    def review_store_file(
        self,
    ) -> Path:

        return (
            self.review_root
            / "review_store.json"
        )

    @property
    def verified_annotations_file(
        self,
    ) -> Path:

        return (
            self.review_root
            / "verified_annotations.json"
        )

    # =====================================================
    # PREDICTIONS
    # =====================================================

    @property
    def prediction_root(
        self,
    ) -> Path:

        return (
            self.project_root
            / "output"
            / "predictions"
            / Path(
                self.selected_folder
            )
        )

    @property
    def predictions_file(
        self,
    ) -> Path:

        return (
            self.prediction_root
            / "predictions.json"
        )

    @property
    def inference_summary_file(
        self,
    ) -> Path:

        return (
            self.prediction_root
            / "inference_summary.json"
        )

    @property
    def preview_directory(
        self,
    ) -> Path:

        return (
            self.prediction_root
            / "previews"
        )

    @property
    def prediction_archive_root(
        self,
    ) -> Path:

        return (
            self.prediction_root
            / "archive"
        )

    # =====================================================
    # MODEL
    # =====================================================

    @property
    def model_root(
        self,
    ) -> Path:

        return (
            self.project_root
            / "models"
            / Path(
                self.selected_folder
            )
        )

    @property
    def model_weights_directory(
        self,
    ) -> Path:

        return (
            self.model_root
            / "weights"
        )

    @property
    def best_model_file(
        self,
    ) -> Path:

        return (
            self.model_weights_directory
            / "best.pt"
        )

    @property
    def last_model_file(
        self,
    ) -> Path:

        return (
            self.model_weights_directory
            / "last.pt"
        )

    @property
    def model_versions_directory(
        self,
    ) -> Path:

        return (
            self.model_root
            / "versions"
        )

    @property
    def champion_report_file(
        self,
    ) -> Path:

        return (
            self.model_root
            / "champion_report.json"
        )

    @property
    def training_report_file(
        self,
    ) -> Path:

        return (
            self.model_root
            / "training_report.json"
        )

    # =====================================================
    # DIRECTORY CREATION
    # =====================================================

    def ensure_working_directories(
        self,
    ) -> None:
        """
        Create only project-generated working directories.

        Original dataset folders/images are not modified.
        """

        directories = [
            self.manual_annotations_directory,
            self.normalized_annotations_directory,
            self.combined_annotations_directory,
            self.training_root,
            self.review_root,
            self.prediction_root,
            self.model_root,
            self.model_weights_directory,
            self.model_versions_directory,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # =====================================================
    # SERIALIZATION
    # =====================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "selected_folder": (
                self.selected_folder
            ),
            "display_name": (
                self.display_name
            ),
            "category_key": (
                self.category_key
            ),
            "mapped_label": (
                self.mapped_label
            ),
            "dataset_root": str(
                self.dataset_root
            ),
            "selected_dataset_path": str(
                self.selected_dataset_path
            ),
            "manual_annotations_directory": str(
                self.manual_annotations_directory
            ),
            "normalized_annotations_file": str(
                self.normalized_annotations_file
            ),
            "combined_annotations_file": str(
                self.combined_annotations_file
            ),
            "training_root": str(
                self.training_root
            ),
            "dataset_yaml": str(
                self.dataset_yaml
            ),
            "review_root": str(
                self.review_root
            ),
            "review_store_file": str(
                self.review_store_file
            ),
            "verified_annotations_file": str(
                self.verified_annotations_file
            ),
            "prediction_root": str(
                self.prediction_root
            ),
            "predictions_file": str(
                self.predictions_file
            ),
            "model_root": str(
                self.model_root
            ),
            "best_model_file": str(
                self.best_model_file
            ),
            "champion_report_file": str(
                self.champion_report_file
            ),
        }


# =========================================================
# LABEL MAPPING HELPERS
# =========================================================

def load_label_mapping(
    mapping_file: str | Path,
) -> dict[str, str]:
    """
    Load folder -> label mapping.

    Supports keys such as:

        S26_2

        S25/S25_2

        S25/Variant_A/Type_2
    """

    mapping_path = Path(
        mapping_file
    ).resolve()

    if not mapping_path.is_file():
        return {}

    try:
        data = json.loads(
            mapping_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:
        raise CategoryContextError(
            "Invalid label mapping JSON: "
            f"{mapping_path}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise CategoryContextError(
            "Label mapping JSON must contain "
            "an object."
        )

    normalized_mapping = {}

    for raw_folder, raw_label in data.items():

        folder = (
            normalize_relative_folder(
                raw_folder
            )
        )

        label = str(
            raw_label
        ).strip()

        if not label:
            continue

        normalized_mapping[
            folder
        ] = label

    return normalized_mapping


def get_mapped_label(
    selected_folder: str | Path,
    mapping: dict[str, str],
) -> str:
    """
    Return exact mapped label for selected folder.
    """

    folder = (
        normalize_relative_folder(
            selected_folder
        )
    )

    label = str(
        mapping.get(
            folder,
            "",
        )
    ).strip()

    if not label:
        raise CategoryContextError(
            "No label mapping found for folder: "
            f"{folder}"
        )

    return label


# =========================================================
# IMAGE-FOLDER DISCOVERY
# =========================================================

def discover_image_folders(
    dataset_root: str | Path,
    supported_extensions: set[str] | None = None,
    ignored_directory_names: set[str] | None = None,
) -> list[str]:
    """
    Recursively discover folders that directly contain
    supported images.

    Example result:

        [
            "S26_2",
            "S25/S25_1",
            "S25/S25_2",
            "S20/Type_A",
        ]

    Parent folders with no direct images are not returned.
    """

    root = Path(
        dataset_root
    ).resolve()

    if not root.is_dir():
        raise CategoryContextError(
            "Dataset root not found: "
            f"{root}"
        )

    if supported_extensions is None:
        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".jfif",
            ".png",
            ".bmp",
            ".webp",
        }

    normalized_extensions = {
        extension.casefold()
        if extension.startswith(".")
        else f".{extension.casefold()}"
        for extension
        in supported_extensions
    }

    if ignored_directory_names is None:
        ignored_directory_names = {
            "image_zip",
            "output",
            "models",
            "data",
            ".venv",
            "__pycache__",
        }

    ignored_names = {
        name.casefold()
        for name
        in ignored_directory_names
    }

    discovered = set()

    for image_path in root.rglob("*"):

        if not image_path.is_file():
            continue

        if (
            image_path.suffix.casefold()
            not in normalized_extensions
        ):
            continue

        try:
            relative_path = (
                image_path.relative_to(
                    root
                )
            )

        except ValueError:
            continue

        parent_parts = (
            relative_path.parts[
                :-1
            ]
        )

        if any(
            part.casefold()
            in ignored_names
            for part
            in parent_parts
        ):
            continue

        parent_folder = (
            relative_path.parent
        )

        if str(parent_folder) == ".":
            continue

        discovered.add(
            parent_folder.as_posix()
        )

    return sorted(
        discovered,
        key=str.casefold,
    )


# =========================================================
# COMMAND-LINE TEST
# =========================================================

def main():
    """
    Basic context test.

    This DOES NOT alter S26_2 model/results.
    """

    parser = argparse.ArgumentParser(
        description="List discovered image folders and configured labels."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Absolute or relative path to the dataset root.",
    )
    args = parser.parse_args()

    project_root = Path(
        __file__
    ).resolve().parent.parent

    dataset_root = Path(
        args.dataset_root
    ).expanduser()

    mapping_file = (
        project_root
        / "config"
        / "label_mapping.json"
    )

    folders = discover_image_folders(
        dataset_root
    )

    print(
        "Image folders discovered:"
    )

    for folder in folders:
        print(
            f"  - {folder}"
        )

    mapping = load_label_mapping(
        mapping_file
    )

    print(
        "\nMapped folders:"
    )

    for folder in folders:

        label = mapping.get(
            folder
        )

        if label:
            print(
                f"  {folder} -> {label}"
            )


if __name__ == "__main__":
    main()