from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
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
    safe_copy_file,
    safe_write_json,
)

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".jfif",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class ModelTrainingError(Exception):
    """Raised when model training or selection fails."""

    pass


class VersionedYOLOTrainer:
    """
    Train and validate a YOLO candidate for one selected
    folder or nested subfolder.

    Every candidate is preserved under:

        models/<selected_folder>/versions/<version>/

    Production model:

        models/<selected_folder>/weights/best.pt

    A candidate replaces the production model only when
    the candidate beats the currently saved champion.

    Comparison priority:

        1. mAP50-95
        2. mAP75
        3. Recall
        4. Precision
        5. mAP50
    """

    def __init__(
        self,
        selected_folder: str,
        mapped_label: str,
        dataset_yaml: str | Path,
        pretrained_model: str | Path,
        model_root: str | Path,
        epochs: int = 20,
        image_size: int = 640,
        batch_size: int = 2,
        device: str = "cpu",
        workers: int = 0,
        seed: int = 42,
        patience: int = 10,
    ):
        self.selected_folder = (
            str(selected_folder)
            .strip()
            .replace(
                "\\",
                "/",
            )
            .strip("/")
        )

        self.mapped_label = str(
            mapped_label
        ).strip()

        self.dataset_yaml = Path(
            dataset_yaml
        ).resolve()

        self.pretrained_model = Path(
            pretrained_model
        ).resolve()

        self.model_root = Path(
            model_root
        ).resolve()

        self.epochs = int(
            epochs
        )

        self.image_size = int(
            image_size
        )

        self.batch_size = int(
            batch_size
        )

        self.device = str(
            device
        ).strip()

        self.workers = int(
            workers
        )

        self.seed = int(
            seed
        )

        self.patience = int(
            patience
        )

        self.versions_root = (
            self.model_root
            / "versions"
        )

        self.production_weights = (
            self.model_root
            / "weights"
        )

        self.champion_report = (
            self.model_root
            / "champion_report.json"
        )

        self.training_report = (
            self.model_root
            / "training_report.json"
        )

        self._validate_configuration()

    # =====================================================
    # CONFIGURATION
    # =====================================================

    def _validate_configuration(
        self,
    ) -> None:
        if not self.selected_folder:
            raise ModelTrainingError(
                "Selected folder is empty."
            )

        if not self.mapped_label:
            raise ModelTrainingError(
                "Mapped label is empty."
            )

        if not self.dataset_yaml.is_file():
            raise ModelTrainingError(
                "dataset.yaml not found: "
                f"{self.dataset_yaml}"
            )

        if not self.pretrained_model.is_file():
            raise ModelTrainingError(
                "Pretrained model not found: "
                f"{self.pretrained_model}"
            )

        if self.epochs <= 0:
            raise ModelTrainingError(
                "epochs must be greater than zero."
            )

        if self.image_size <= 0:
            raise ModelTrainingError(
                "image_size must be greater than zero."
            )

        if self.batch_size <= 0:
            raise ModelTrainingError(
                "batch_size must be greater than zero."
            )

        if self.workers < 0:
            raise ModelTrainingError(
                "workers cannot be negative."
            )

        if self.patience < 0:
            raise ModelTrainingError(
                "patience cannot be negative."
            )

        self._validate_training_dataset()

        self.versions_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.production_weights.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _validate_training_dataset(
        self,
    ) -> None:
        split_counts: dict[str, int] = {}

        for split in ("train", "val"):
            images_directory = (
                self.dataset_yaml.parent
                / "images"
                / split
            )
            labels_directory = (
                self.dataset_yaml.parent
                / "labels"
                / split
            )

            if not images_directory.is_dir():
                raise ModelTrainingError(
                    f"{split} images directory not found: "
                    f"{images_directory}"
                )

            if not labels_directory.is_dir():
                raise ModelTrainingError(
                    f"{split} labels directory not found: "
                    f"{labels_directory}"
                )

            images = [
                path
                for path in images_directory.iterdir()
                if (
                    path.is_file()
                    and path.suffix.casefold()
                    in IMAGE_EXTENSIONS
                )
            ]
            labels = [
                path
                for path in labels_directory.iterdir()
                if (
                    path.is_file()
                    and path.suffix.casefold() == ".txt"
                )
            ]

            if not images:
                raise ModelTrainingError(
                    f"No {split} images found in "
                    f"{images_directory}"
                )

            if not labels:
                raise ModelTrainingError(
                    f"No {split} labels found in "
                    f"{labels_directory}"
                )

            image_stems = [
                path.stem.casefold()
                for path in images
            ]
            label_stems = [
                path.stem.casefold()
                for path in labels
            ]

            if len(image_stems) != len(set(image_stems)):
                raise ModelTrainingError(
                    f"Duplicate image basenames found in "
                    f"{images_directory}"
                )

            if len(label_stems) != len(set(label_stems)):
                raise ModelTrainingError(
                    f"Duplicate label basenames found in "
                    f"{labels_directory}"
                )

            if set(image_stems) != set(label_stems):
                missing_labels = sorted(
                    set(image_stems) - set(label_stems)
                )
                orphan_labels = sorted(
                    set(label_stems) - set(image_stems)
                )
                raise ModelTrainingError(
                    f"{split} images and labels do not match. "
                    f"Missing labels: {missing_labels}; "
                    f"orphan labels: {orphan_labels}."
                )

            split_counts[split] = len(images)

        manifest_path = (
            self.dataset_yaml.parent
            / "dataset_manifest.json"
        )
        if not manifest_path.exists():
            return

        manifest = self._load_json(
            manifest_path
        )
        if manifest is None:
            raise ModelTrainingError(
                "Dataset manifest exists but is invalid: "
                f"{manifest_path}"
            )

        expected_counts = {
            "total_images": (
                split_counts["train"]
                + split_counts["val"]
            ),
            "training_images": split_counts["train"],
            "validation_images": split_counts["val"],
        }

        for field, actual_count in expected_counts.items():
            manifest_count = manifest.get(field)
            if (
                isinstance(manifest_count, bool)
                or not isinstance(manifest_count, int)
                or manifest_count != actual_count
            ):
                raise ModelTrainingError(
                    "Dataset manifest count does not match "
                    f"the generated files for '{field}': "
                    f"manifest={manifest_count!r}, "
                    f"files={actual_count}."
                )

    # =====================================================
    # JSON HELPERS
    # =====================================================

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any] | None:
        if not path.is_file():
            return None

        try:
            data = json.loads(
                path.read_text(
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
    def _write_json_atomic(
        destination: Path,
        payload: dict[str, Any],
    ) -> None:
        try:
            safe_write_json(
                destination=destination,
                payload=payload,
                retries=30,
            )

        except SafeIOError as exc:
            raise ModelTrainingError(
                "Could not safely write JSON: "
                f"{destination}: {exc}"
            ) from exc

    # =====================================================
    # DATASET INFORMATION
    # =====================================================

    def _load_dataset_manifest(
        self,
    ) -> dict[str, Any] | None:
        manifest_path = (
            self.dataset_yaml.parent
            / "dataset_manifest.json"
        )

        return self._load_json(
            manifest_path
        )

    def _dataset_information(
        self,
    ) -> dict[str, Any]:
        manifest = (
            self._load_dataset_manifest()
        )

        if manifest is None:
            return {
                "total_images": None,
                "total_boxes": None,
                "training_images": None,
                "validation_images": None,
                "manifest": None,
            }

        manifest_folder = str(
            manifest.get(
                "selected_folder",
                "",
            )
        ).strip()

        if (
            manifest_folder
            and manifest_folder
            != self.selected_folder
        ):
            raise ModelTrainingError(
                "Training dataset belongs to "
                f"'{manifest_folder}', but selected "
                f"folder is '{self.selected_folder}'."
            )

        classes = manifest.get(
            "classes",
            {},
        )

        if isinstance(
            classes,
            dict,
        ):
            available_classes = {
                str(name).strip()
                for name
                in classes.keys()
            }

            if (
                available_classes
                and self.mapped_label
                not in available_classes
            ):
                raise ModelTrainingError(
                    "Mapped label is not present in "
                    "the dataset manifest: "
                    f"{self.mapped_label}"
                )

        return {
            "total_images": (
                manifest.get(
                    "total_images"
                )
            ),
            "total_boxes": (
                manifest.get(
                    "total_bounding_boxes"
                )
            ),
            "training_images": (
                manifest.get(
                    "training_images"
                )
            ),
            "validation_images": (
                manifest.get(
                    "validation_images"
                )
            ),
            "manifest": str(
                self.dataset_yaml.parent
                / "dataset_manifest.json"
            ),
        }

    # =====================================================
    # VERSION NAME
    # =====================================================

    @staticmethod
    def _safe_version_prefix(
        selected_folder: str,
    ) -> str:
        prefix = (
            selected_folder
            .replace(
                "\\",
                "__",
            )
            .replace(
                "/",
                "__",
            )
            .replace(
                " ",
                "_",
            )
            .strip("_")
            .casefold()
        )

        return (
            prefix
            or "selected_folder"
        )

    def _make_version_name(
        self,
        total_images: int | None,
    ) -> str:
        """
        Create a short Windows-safe version name.

        The selected folder is already represented by
        model_root, so repeating the complete nested
        folder path in the version name is unnecessary.

        Example:

            v0022_20261030_135852
        """

        if total_images is None:
            image_count = "unk"

        else:
            image_count = (
                f"{int(total_images):04d}"
            )

        timestamp = (
            datetime.now()
            .astimezone()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        return (
            f"v{image_count}_"
            f"{timestamp}"
        )


    # =====================================================
    # METRIC EXTRACTION
    # =====================================================

    @staticmethod
    def _mean(
        values: Any,
    ) -> float:
        try:
            if hasattr(
                values,
                "mean",
            ):
                return float(
                    values.mean()
                )

            sequence = list(
                values
            )

            if not sequence:
                return 0.0

            return float(
                sum(
                    sequence
                )
                / len(
                    sequence
                )
            )

        except Exception:
            return 0.0

    @classmethod
    def _extract_metrics(
        cls,
        validation_result: Any,
    ) -> dict[str, float]:
        box_metrics = getattr(
            validation_result,
            "box",
            None,
        )

        if box_metrics is None:
            raise ModelTrainingError(
                "Validation did not return "
                "bounding-box metrics."
            )

        precision = float(
            getattr(
                box_metrics,
                "mp",
                0.0,
            )
        )

        recall = float(
            getattr(
                box_metrics,
                "mr",
                0.0,
            )
        )

        map50 = float(
            getattr(
                box_metrics,
                "map50",
                0.0,
            )
        )

        map50_95 = float(
            getattr(
                box_metrics,
                "map",
                0.0,
            )
        )

        map75 = 0.0

        all_ap = getattr(
            box_metrics,
            "all_ap",
            None,
        )

        if all_ap is not None:
            try:
                # IoU columns:
                # 0.50, 0.55, 0.60, 0.65, 0.70,
                # 0.75, 0.80, 0.85, 0.90, 0.95

                map75 = cls._mean(
                    all_ap[
                        :,
                        5,
                    ]
                )

            except Exception:
                try:
                    rows = list(
                        all_ap
                    )

                    map75 = cls._mean(
                        row[5]
                        for row
                        in rows
                    )

                except Exception:
                    map75 = 0.0

        return {
            "precision": (
                precision
            ),
            "recall": (
                recall
            ),
            "map50": (
                map50
            ),
            "map75": (
                map75
            ),
            "map50_95": (
                map50_95
            ),
        }

    # =====================================================
    # CHAMPION COMPARISON
    # =====================================================

    @staticmethod
    def _score(
        metrics: dict[str, Any],
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
    ]:
        return (
            float(
                metrics.get(
                    "map50_95",
                    0.0,
                )
            ),
            float(
                metrics.get(
                    "map75",
                    0.0,
                )
            ),
            float(
                metrics.get(
                    "recall",
                    0.0,
                )
            ),
            float(
                metrics.get(
                    "precision",
                    0.0,
                )
            ),
            float(
                metrics.get(
                    "map50",
                    0.0,
                )
            ),
        )

    def _load_champion(
        self,
    ) -> dict[str, Any] | None:
        return self._load_json(
            self.champion_report
        )

    def _candidate_is_better(
        self,
        candidate_metrics: dict[
            str,
            float,
        ],
        champion: dict[
            str,
            Any,
        ] | None,
    ) -> tuple[
        bool,
        str,
    ]:
        if champion is None:
            return (
                True,
                (
                    "No champion report existed; "
                    "candidate became initial "
                    "production model."
                ),
            )

        champion_folder = str(
            champion.get(
                "selected_folder",
                self.selected_folder,
            )
        ).strip()

        if (
            champion_folder
            != self.selected_folder
        ):
            raise ModelTrainingError(
                "Champion report belongs to "
                f"'{champion_folder}', not "
                f"'{self.selected_folder}'."
            )

        champion_metrics = (
            champion.get(
                "validation_metrics"
            )
        )

        if not isinstance(
            champion_metrics,
            dict,
        ):
            return (
                True,
                (
                    "Champion metrics were missing; "
                    "candidate became production model."
                ),
            )

        candidate_score = (
            self._score(
                candidate_metrics
            )
        )

        champion_score = (
            self._score(
                champion_metrics
            )
        )

        if (
            candidate_score
            > champion_score
        ):
            return (
                True,
                (
                    "Candidate score is better "
                    "than saved champion score."
                ),
            )

        return (
            False,
            (
                "Candidate did not beat "
                "the saved champion score."
            ),
        )

    # =====================================================
    # SAFE MODEL COPY
    # =====================================================

    @staticmethod
    def _copy_atomic(
        source: Path,
        destination: Path,
    ) -> None:
        try:
            safe_copy_file(
                source=source,
                destination=destination,
                retries=40,
            )

        except SafeIOError as exc:
            raise ModelTrainingError(
                "Could not safely copy model file: "
                f"{source} -> {destination}: "
                f"{exc}"
            ) from exc

    # =====================================================
    # PROMOTE CANDIDATE
    # =====================================================

    def _promote_candidate(
        self,
        candidate_best: Path,
        candidate_last: Path,
        champion_payload: dict[
            str,
            Any,
        ],
    ) -> None:
        production_best = (
            self.production_weights
            / "best.pt"
        )

        production_last = (
            self.production_weights
            / "last.pt"
        )

        self._copy_atomic(
            candidate_best,
            production_best,
        )

        self._copy_atomic(
            candidate_last,
            production_last,
        )

        self._write_json_atomic(
            self.champion_report,
            champion_payload,
        )

    # =====================================================
    # TRAIN
    # =====================================================

    def train(
        self,
    ) -> dict[str, Any]:
        try:
            from ultralytics import YOLO

        except ImportError as exc:
            raise ModelTrainingError(
                "Ultralytics is not installed."
            ) from exc

        dataset_information = (
            self._dataset_information()
        )

        total_images = (
            dataset_information[
                "total_images"
            ]
        )

        version_name = (
            self._make_version_name(
                total_images
            )
        )

        version_directory = (
            self.versions_root
            / version_name
        )

        runs_directory = (
            version_directory
            / "runs"
        )

        version_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        print(
            "Selected folder: "
            f"{self.selected_folder}"
        )

        print(
            "Mapped label: "
            f"{self.mapped_label}"
        )

        print(
            "Training dataset:"
        )

        print(
            f"  - {self.dataset_yaml}"
        )

        print(
            "Model root:"
        )

        print(
            f"  - {self.model_root}"
        )

        print(
            "Training candidate version:"
        )

        print(
            f"  - {version_name}"
        )

        # -------------------------------------------------
        # LOAD PRETRAINED MODEL
        # -------------------------------------------------

        model = YOLO(
            str(
                self.pretrained_model
            )
        )

        # -------------------------------------------------
        # TRAINING
        # -------------------------------------------------

        training_result = model.train(
            data=str(
                self.dataset_yaml
            ),
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            device=self.device,
            workers=self.workers,
            seed=self.seed,
            deterministic=True,
            patience=self.patience,
            project=str(
                runs_directory
            ),
            name="training",
            exist_ok=False,
            pretrained=True,
            val=True,
            verbose=True,
        )

        default_save_directory = (
            runs_directory
            / "training"
        )

        save_directory_value = getattr(
            training_result,
            "save_dir",
            default_save_directory,
        )

        save_directory = Path(
            str(
                save_directory_value
            )
        ).resolve()

        candidate_best_source = (
            save_directory
            / "weights"
            / "best.pt"
        )

        candidate_last_source = (
            save_directory
            / "weights"
            / "last.pt"
        )

        if not candidate_best_source.is_file():
            raise ModelTrainingError(
                "Candidate best.pt was not created: "
                f"{candidate_best_source}"
            )

        if not candidate_last_source.is_file():
            raise ModelTrainingError(
                "Candidate last.pt was not created: "
                f"{candidate_last_source}"
            )

        # -------------------------------------------------
        # PRESERVE CANDIDATE
        # -------------------------------------------------

        candidate_weights_directory = (
            version_directory
            / "weights"
        )

        candidate_best = (
            candidate_weights_directory
            / "best.pt"
        )

        candidate_last = (
            candidate_weights_directory
            / "last.pt"
        )

        self._copy_atomic(
            candidate_best_source,
            candidate_best,
        )

        self._copy_atomic(
            candidate_last_source,
            candidate_last,
        )

        # -------------------------------------------------
        # VALIDATE CANDIDATE
        # -------------------------------------------------

        validation_model = YOLO(
            str(
                candidate_best
            )
        )

        validation_result = (
            validation_model.val(
                data=str(
                    self.dataset_yaml
                ),
                imgsz=self.image_size,
                batch=self.batch_size,
                device=self.device,
                workers=self.workers,
                split="val",
                project=str(
                    runs_directory
                ),
                name="validation",
                exist_ok=False,
                verbose=True,
            )
        )

        validation_metrics = (
            self._extract_metrics(
                validation_result
            )
        )

        # -------------------------------------------------
        # CHAMPION DECISION
        # -------------------------------------------------

        previous_champion = (
            self._load_champion()
        )

        (
            production_promoted,
            decision_reason,
        ) = self._candidate_is_better(
            validation_metrics,
            previous_champion,
        )

        created_at = (
            datetime.now()
            .astimezone()
            .isoformat()
        )

        candidate_report = {
            "status": "completed",
            "selected_folder": (
                self.selected_folder
            ),
            "mapped_label": (
                self.mapped_label
            ),
            "version": (
                version_name
            ),
            "created_at": (
                created_at
            ),
            "device": (
                self.device
            ),
            "dataset_yaml": str(
                self.dataset_yaml
            ),
            "dataset_manifest": (
                dataset_information[
                    "manifest"
                ]
            ),
            "dataset_images": (
                dataset_information[
                    "total_images"
                ]
            ),
            "dataset_boxes": (
                dataset_information[
                    "total_boxes"
                ]
            ),
            "training_images": (
                dataset_information[
                    "training_images"
                ]
            ),
            "validation_images": (
                dataset_information[
                    "validation_images"
                ]
            ),
            "pretrained_model": str(
                self.pretrained_model
            ),
            "candidate_directory": str(
                version_directory
            ),
            "candidate_best_weights": str(
                candidate_best
            ),
            "candidate_last_weights": str(
                candidate_last
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "production_promoted": (
                production_promoted
            ),
            "decision_reason": (
                decision_reason
            ),
        }

        candidate_report_file = (
            version_directory
            / "training_report.json"
        )

        self._write_json_atomic(
            candidate_report_file,
            candidate_report,
        )

        # -------------------------------------------------
        # PROMOTION
        # -------------------------------------------------

        if production_promoted:
            champion_payload = {
                **candidate_report,
                "production_best_weights": str(
                    self.production_weights
                    / "best.pt"
                ),
                "production_last_weights": str(
                    self.production_weights
                    / "last.pt"
                ),
            }

            self._promote_candidate(
                candidate_best=(
                    candidate_best
                ),
                candidate_last=(
                    candidate_last
                ),
                champion_payload=(
                    champion_payload
                ),
            )

        # -------------------------------------------------
        # ACTIVE CHAMPION
        # -------------------------------------------------

        active_champion = (
            self._load_champion()
        )

        production_best = (
            self.production_weights
            / "best.pt"
        )

        production_last = (
            self.production_weights
            / "last.pt"
        )

        final_report = {
            **candidate_report,
            "production_best_weights": (
                str(
                    production_best
                )
                if production_best.is_file()
                else None
            ),
            "production_last_weights": (
                str(
                    production_last
                )
                if production_last.is_file()
                else None
            ),
            "active_champion_version": (
                active_champion.get(
                    "version"
                )
                if isinstance(
                    active_champion,
                    dict,
                )
                else None
            ),
            "active_champion_metrics": (
                active_champion.get(
                    "validation_metrics"
                )
                if isinstance(
                    active_champion,
                    dict,
                )
                else None
            ),
        }

        self._write_json_atomic(
            self.training_report,
            final_report,
        )

        return final_report


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
# PRETRAINED MODEL
# =========================================================

def get_pretrained_model(
    project_root: Path,
) -> Path:
    """
    Resolve the local pretrained model.

    Primary expected file:

        models/pretrained/yolo26n.pt
    """

    primary_model = (
        project_root
        / "models"
        / "pretrained"
        / "yolo26n.pt"
    )

    if primary_model.is_file():
        return primary_model

    alternative_models = [
        (
            project_root
            / "models"
            / "pretrained"
            / "yolo11n.pt"
        ),
        (
            project_root
            / "models"
            / "pretrained"
            / "yolov8n.pt"
        ),
    ]

    for candidate in alternative_models:
        if candidate.is_file():
            return candidate

    raise ModelTrainingError(
        "No local pretrained model was found under: "
        f"{project_root / 'models' / 'pretrained'}"
    )


# =========================================================
# CLI
# =========================================================

def main() -> None:
    """
    Examples:

        python -m src.model_trainer --folder S26_2 --dataset-root C:/path/to/dataset

        python -m src.model_trainer --folder S25 --dataset-root C:/path/to/dataset

        python -m src.model_trainer --folder S16/S16_2 --dataset-root C:/path/to/dataset
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train a versioned YOLO candidate "
            "for a selected folder/subfolder."
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

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs. Default: 20",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="YOLO image size. Default: 640",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size. Default: 2",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device. Default: cpu",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Data-loader workers. Default: 0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience. Default: 10",
    )

    args = parser.parse_args()

    try:
        context = build_context(
            args.folder,
            args.dataset_root,
        )

    except CategoryContextError as exc:
        raise ModelTrainingError(
            str(
                exc
            )
        ) from exc

    context.ensure_working_directories()

    pretrained_model = (
        get_pretrained_model(
            context.project_root
        )
    )

    trainer = VersionedYOLOTrainer(
        selected_folder=(
            context.selected_folder
        ),
        mapped_label=(
            context.mapped_label
        ),
        dataset_yaml=(
            context.dataset_yaml
        ),
        pretrained_model=(
            pretrained_model
        ),
        model_root=(
            context.model_root
        ),
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
    )

    result = trainer.train()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()