from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from PIL import Image, ImageDraw

from src.annotation_importer import MakeSenseCSVImporter
from src.category_context import (
    CategoryContext,
    CategoryContextError,
    discover_image_folders,
    normalize_relative_folder,
)
from src.dataset_scanner import DatasetScanner
from src.drawable_editor import (
    DrawableEditorError,
    show_drawable_bbox_editor,
)
from src.label_registry import LabelRegistry
from src.review_manager import AnnotationReviewManager
from src.duplicate_ui import display_duplicate_management
from src.rename_ui import display_image_rename
from src.collector_ui import display_image_collector
from src.similar_image_ui import display_similar_image_search


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Ericsson Label Automation",
    page_icon="🏷️",
    layout="wide",
)

st.title(
    "Ericsson Image Labelling Automation"
)

st.caption(
    "Local dataset validation, duplicate analysis, "
    "dynamic folder selection, annotation import "
    "and AI prediction review"
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

CONFIG_PATH = (
    PROJECT_ROOT
    / "config.yaml"
)


# =========================================================
# CONFIGURATION HELPERS
# =========================================================

def load_config() -> dict:
    """
    Load config.yaml.
    """

    if not CONFIG_PATH.is_file():
        return {}

    return (
        yaml.safe_load(
            CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )
        or {}
    )


def normalize_path(
    value,
) -> Path:
    """
    Normalize a user-entered path.
    """

    return Path(
        str(value)
        .strip()
        .strip('"')
    ).expanduser()


def get_mapping_path(
    config: dict,
) -> Path:
    """
    Resolve label mapping file path.
    """

    path = Path(
        config.get(
            "mapping_file",
            "config/label_mapping.json",
        )
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    )


def normalize_mapping(
    mapping: dict,
) -> dict[str, str]:
    """
    Normalize folder mapping keys.

    Example:

        S26_2
        S16/S16_1
        S25/Subfolder
    """

    result = {}

    for folder, label in mapping.items():
        try:
            key = normalize_relative_folder(
                folder
            )

        except Exception:
            key = (
                str(folder)
                .replace(
                    "\\",
                    "/",
                )
                .strip("/")
            )

        value = str(
            label
        ).strip()

        if key and value:
            result[
                key
            ] = value

    return result


# =========================================================
# SESSION STATE
# =========================================================

def initialize_state(
    config: dict,
) -> None:
    """
    Initialize Streamlit state.
    """

    defaults = {
        "report": None,
        "labels": [],
        "mapping": {},
        "annotation_import_result": None,
        "annotation_json_path": None,
        "unique_zip_path": None,
        "dataset_root_value": "",
        "label_filename_value": config.get(
            "label_file",
            "ericson_classification label.txt",
        ),
        "selected_working_folder": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[
                key
            ] = value


# =========================================================
# DATASET SCANNING
# =========================================================


def perform_scan(
    config: dict,
    mapping_path: Path,
) -> int:
    """
    Scan images and detect duplicates.

    The master-label file is optional during scanning.
    Labels are required only for folder mapping,
    annotation import, training and AI automation.
    """

    dataset_root = normalize_path(
        st.session_state.dataset_root_value
    )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "Dataset root not found: "
            f"{dataset_root}"
        )

    scanner = DatasetScanner(
        dataset_root,
        config.get(
            "supported_extensions"
        ),
    )

    report = scanner.scan()

    st.session_state.report = report

    st.session_state[
        "annotation_import_result"
    ] = None

    st.session_state[
        "annotation_json_path"
    ] = None

    st.session_state[
        "unique_zip_path"
    ] = None

    label_filename = str(
        st.session_state.get(
            "label_filename_value",
            "",
        )
    ).strip()

    label_file = (
        dataset_root
        / label_filename
    )

    if (
        label_filename
        and label_file.is_file()
    ):
        try:
            registry = LabelRegistry(
                label_file,
                mapping_path,
            ).load()

            st.session_state.labels = (
                registry.labels
            )

            st.session_state.mapping = (
                normalize_mapping(
                    registry.mapping
                )
            )

        except Exception as exc:
            st.session_state.labels = []

            st.session_state.mapping = {}

            st.warning(
                "Image and duplicate scan completed, "
                "but label configuration could not "
                f"be loaded: {exc}"
            )

    else:
        st.session_state.labels = []

        st.session_state.mapping = {}

    folders = discover_image_folders(
        dataset_root
    )

    current_folder = (
        st.session_state.get(
            "selected_working_folder"
        )
    )

    if current_folder not in folders:
        if "S26_2" in folders:
            st.session_state[
                "selected_working_folder"
            ] = "S26_2"

        elif folders:
            st.session_state[
                "selected_working_folder"
            ] = folders[0]

        else:
            st.session_state[
                "selected_working_folder"
            ] = None

    return len(
        report.get(
            "records",
            [],
        )
    )


# =========================================================
# UNIQUE IMAGE CALCULATION
# =========================================================

def calculate_unique_image_summary(
    report: dict,
):
    """
    Keep one valid image for each SHA-256 hash.
    """

    valid_records = [
        record
        for record
        in report["records"]
        if (
            record["status"] == "ok"
            and record["sha256"]
        )
    ]

    corrupt_records = [
        record
        for record
        in report["records"]
        if (
            record["status"]
            == "corrupt"
        )
    ]

    seen_hashes = set()

    unique_records = []

    excluded_records = []

    for record in valid_records:
        file_hash = (
            record["sha256"]
        )

        if file_hash in seen_hashes:
            excluded_records.append(
                record
            )

            continue

        seen_hashes.add(
            file_hash
        )

        unique_records.append(
            record
        )

    summary = {
        "total_images": len(
            report["records"]
        ),
        "valid_images": len(
            valid_records
        ),
        "corrupt_images": len(
            corrupt_records
        ),
        "unique_images": len(
            unique_records
        ),
        "repeated_files": len(
            excluded_records
        ),
        "duplicate_groups": len(
            report["duplicates"]
        ),
    }

    return (
        unique_records,
        excluded_records,
        corrupt_records,
        summary,
    )


# =========================================================
# UNIQUE ZIP
# =========================================================

def create_unique_images_zip(
    report: dict,
    progress_callback=None,
) -> Path:
    """
    Create ZIP containing one image per
    unique SHA-256 hash.

    Original dataset is not modified.
    """

    dataset_root = normalize_path(
        report["root"]
    )

    (
        unique_records,
        excluded_records,
        corrupt_records,
        summary,
    ) = calculate_unique_image_summary(
        report
    )

    if not unique_records:
        raise ValueError(
            "No valid unique images were found."
        )

    output_directory = (
        dataset_root
        / "image_zip"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_path = (
        output_directory
        / "unique_images.zip"
    )

    temporary_path = (
        output_directory
        / "unique_images.tmp.zip"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    included_images = []

    missing_images = []

    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=(
                zipfile.ZIP_DEFLATED
            ),
            compresslevel=6,
            allowZip64=True,
        ) as archive:

            total = len(
                unique_records
            )

            for index, record in enumerate(
                unique_records,
                start=1,
            ):
                source_path = Path(
                    record[
                        "path"
                    ]
                )

                relative_path = Path(
                    record[
                        "relative_path"
                    ]
                )

                if not source_path.is_file():
                    missing_images.append(
                        record[
                            "relative_path"
                        ]
                    )

                    continue

                archive.write(
                    source_path,
                    arcname=(
                        relative_path
                        .as_posix()
                    ),
                )

                included_images.append(
                    record[
                        "relative_path"
                    ]
                )

                if progress_callback:
                    progress_callback(
                        index,
                        total,
                        record[
                            "relative_path"
                        ],
                    )

            manifest = {
                "description": (
                    "Deduplicated Ericsson "
                    "image dataset"
                ),
                "duplicate_rule": (
                    "One file retained for each "
                    "unique SHA-256 hash"
                ),
                "original_dataset_modified": (
                    False
                ),
                "summary": {
                    **summary,
                    "images_written": len(
                        included_images
                    ),
                    "missing_images": len(
                        missing_images
                    ),
                },
                "included_images": (
                    included_images
                ),
                "excluded_duplicate_files": [
                    record[
                        "relative_path"
                    ]
                    for record
                    in excluded_records
                ],
                "corrupt_files_skipped": [
                    record[
                        "relative_path"
                    ]
                    for record
                    in corrupt_records
                ],
                "missing_files_skipped": (
                    missing_images
                ),
            }

            archive.writestr(
                "unique_images_manifest.json",
                json.dumps(
                    manifest,
                    indent=2,
                    ensure_ascii=False,
                ),
            )

        os.replace(
            temporary_path,
            final_path,
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise

    return final_path


# =========================================================
# ANNOTATION PREVIEW
# =========================================================

def draw_annotations(
    dataset_root,
    image_record,
    color=(255, 165, 0),
    include_confidence=True,
):
    """
    Draw annotation boxes on image copy.
    """

    image_path = (
        normalize_path(
            dataset_root
        )
        / image_record[
            "image_path"
        ]
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            "Image not found: "
            f"{image_path}"
        )

    with Image.open(
        image_path
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

    for annotation in image_record.get(
        "annotations",
        [],
    ):
        box = [
            int(
                round(
                    float(
                        annotation[
                            name
                        ]
                    )
                )
            )
            for name
            in (
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            )
        ]

        label = str(
            annotation.get(
                "class_name",
                "",
            )
        )

        confidence = annotation.get(
            "confidence"
        )

        if (
            include_confidence
            and confidence is not None
        ):
            label = (
                f"{label} "
                f"{float(confidence):.2f}"
            )

        draw.rectangle(
            box,
            outline=color,
            width=line_width,
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

        text_color = (
            (0, 0, 0)
            if color
            == (255, 165, 0)
            else (255, 255, 255)
        )

        draw.text(
            text_position,
            label,
            fill=text_color,
        )

    return preview


# =========================================================
# DYNAMIC / LEGACY PATHS
# =========================================================

def legacy_or_dynamic_paths(
    context: CategoryContext,
) -> dict[str, Path]:
    """
    Preserve existing S26_2 annotations without
    moving or deleting them.

    New folders use dynamic CategoryContext paths.
    """

    if (
        context.selected_folder
        == "S26_2"
    ):
        legacy_normalized = (
            PROJECT_ROOT
            / "data"
            / "normalized_annotations"
            / "s26_2_manual_annotations.json"
        )

        if legacy_normalized.exists():
            normalized_file = (
                legacy_normalized
            )

        else:
            normalized_file = (
                context
                .normalized_annotations_file
            )

    else:
        normalized_file = (
            context
            .normalized_annotations_file
        )

    return {
        "manual_dir": (
            context
            .manual_annotations_directory
        ),
        "normalized": (
            normalized_file
        ),
        "predictions": (
            context
            .predictions_file
        ),
        "review_store": (
            context
            .review_store_file
        ),
        "verified": (
            context
            .verified_annotations_file
        ),
    }


# =========================================================
# CSV DISCOVERY
# =========================================================

def discover_annotation_csv_files(
    manual_directory: Path,
) -> list[Path]:
    """
    Discover CSV files only for selected folder.
    """

    if not manual_directory.is_dir():
        return []

    return sorted(
        manual_directory.rglob(
            "*.csv"
        )
    )


# =========================================================
# ANNOTATION IMPORT TAB
# =========================================================

@st.fragment
def display_annotation_import_tab(
    report: dict,
    context: CategoryContext,
    paths: dict[str, Path],
):
    st.subheader(
        "Multiple CSV Annotation Import"
    )

    st.caption(
        f"Working folder: "
        f"{context.display_name} | "
        f"Label: {context.mapped_label}"
    )

    paths[
        "manual_dir"
    ].mkdir(
        parents=True,
        exist_ok=True,
    )

    available_files = (
        discover_annotation_csv_files(
            paths[
                "manual_dir"
            ]
        )
    )

    if not available_files:
        st.warning(
            "No CSV files found under: "
            f"{paths['manual_dir']}"
        )

        return

    csv_options = {
        path.relative_to(
            PROJECT_ROOT
        ).as_posix(): path
        for path
        in available_files
    }

    selected_names = st.multiselect(
        "Select annotation CSV files",
        options=list(
            csv_options.keys()
        ),
        default=list(
            csv_options.keys()
        ),
        key=(
            "multi_csv_selection_"
            + context.category_key
        ),
    )

    if st.button(
        "Import and Merge Selected CSV Files",
        type="primary",
        key=(
            "merge_csv_"
            + context.category_key
        ),
    ):
        if not selected_names:
            st.error(
                "Select at least one CSV file."
            )

        else:
            try:
                selected_paths = [
                    csv_options[
                        name
                    ]
                    for name
                    in selected_names
                ]

                importer = (
                    MakeSenseCSVImporter(
                        dataset_root=(
                            report["root"]
                        ),
                        valid_labels=[
                            context.mapped_label
                        ],
                        selected_folder=(
                            context.selected_folder
                        ),
                    )
                )

                result = (
                    importer
                    .import_multiple_csv(
                        selected_paths
                    )
                )

                saved_path = (
                    importer
                    .save_normalized_json(
                        import_result=result,
                        output_file=(
                            paths[
                                "normalized"
                            ]
                        ),
                    )
                )

                st.session_state[
                    "annotation_import_result"
                ] = result

                st.session_state[
                    "annotation_json_path"
                ] = str(
                    saved_path
                )

                st.success(
                    "CSV files imported and merged."
                )

            except Exception as exc:
                st.error(
                    f"Import failed: {exc}"
                )

    result = st.session_state.get(
        "annotation_import_result"
    )

    if not result:
        return

    columns = st.columns(
        4
    )

    columns[0].metric(
        "CSV files",
        result.get(
            "source_file_count",
            0,
        ),
    )

    columns[1].metric(
        "Images",
        result.get(
            "imported_images",
            0,
        ),
    )

    columns[2].metric(
        "Boxes",
        result.get(
            "imported_annotations",
            0,
        ),
    )

    columns[3].metric(
        "Duplicates removed",
        result.get(
            "duplicate_annotations_removed",
            0,
        ),
    )

    if result.get(
        "errors"
    ):
        st.error(
            "Annotation import contains errors."
        )

        st.dataframe(
            pd.DataFrame(
                result[
                    "errors"
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    imported_images = (
        result.get(
            "images",
            [],
        )
    )

    if imported_images:
        image_lookup = {
            item[
                "image_path"
            ]: item
            for item
            in imported_images
        }

        selected_image = (
            st.selectbox(
                "Preview imported annotation",
                options=list(
                    image_lookup.keys()
                ),
                key=(
                    "import_preview_"
                    + context.category_key
                ),
            )
        )

        try:
            preview = (
                draw_annotations(
                    dataset_root=(
                        report["root"]
                    ),
                    image_record=(
                        image_lookup[
                            selected_image
                        ]
                    ),
                    color=(
                        255,
                        0,
                        0,
                    ),
                    include_confidence=False,
                )
            )

            st.image(
                preview,
                caption=(
                    selected_image
                ),
                width="stretch",
            )

        except Exception as exc:
            st.error(
                f"Preview failed: {exc}"
            )

    if paths[
        "normalized"
    ].is_file():

        st.download_button(
            "Download Combined Annotation JSON",
            data=(
                paths[
                    "normalized"
                ].read_bytes()
            ),
            file_name=(
                context.category_key
                + "_manual_annotations.json"
            ),
            mime="application/json",
            key=(
                "download_annotation_"
                + context.category_key
            ),
        )


# =========================================================
# REVIEW MANAGER
# =========================================================

def create_review_manager(
    report: dict,
    context: CategoryContext,
    paths: dict[str, Path],
) -> AnnotationReviewManager:
    """
    Create review manager for selected folder.
    """

    paths[
        "review_store"
    ].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return AnnotationReviewManager(
        dataset_root=(
            report["root"]
        ),
        predictions_json=(
            paths[
                "predictions"
            ]
        ),
        output_file=(
            paths[
                "review_store"
            ]
        ),
        valid_labels=[
            context.mapped_label
        ],
    )


# =========================================================
# AI REVIEW TAB
# =========================================================

@st.fragment
def display_ai_review_tab(
    report: dict,
    context: CategoryContext,
    paths: dict[str, Path],
):
    st.subheader(
        "AI Prediction Review"
    )

    st.caption(
        f"Working folder: "
        f"{context.display_name} | "
        f"Label: {context.mapped_label}"
    )

    predictions_file = (
        paths[
            "predictions"
        ]
    )

    if not predictions_file.is_file():
        st.warning(
            "Predictions not found for "
            f"{context.display_name}: "
            f"{predictions_file}"
        )

        return

    try:
        manager = (
            create_review_manager(
                report,
                context,
                paths,
            )
        )

    except Exception as exc:
        st.error(
            "Could not open review data: "
            f"{exc}"
        )

        return

    summary = (
        manager.get_summary()
    )

    columns = st.columns(
        6
    )

    columns[0].metric(
        "Total",
        summary[
            "total_images"
        ],
    )

    columns[1].metric(
        "Pending",
        summary[
            "pending"
        ],
    )

    columns[2].metric(
        "Verified",
        summary[
            "verified"
        ],
    )

    columns[3].metric(
        "Corrected",
        summary[
            "corrected"
        ],
    )

    columns[4].metric(
        "Rejected",
        summary[
            "rejected"
        ],
    )

    columns[5].metric(
        "No Object",
        summary[
            "no_object"
        ],
    )

    status_filter = (
        st.selectbox(
            "Review-status filter",
            options=[
                "all",
                "pending",
                "verified",
                "corrected",
                "rejected",
                "no_object",
            ],
            key=(
                "review_status_"
                + context.category_key
            ),
        )
    )

    queue_status = (
        None
        if status_filter == "all"
        else status_filter
    )

    queue = (
        manager.get_review_queue(
            status=queue_status,
            lowest_confidence_first=True,
        )
    )

    if not queue:
        st.info(
            "No images match this "
            "review status."
        )

        return

    image_lookup = {
        record[
            "image_path"
        ]: record
        for record
        in queue
    }

    selected_image_path = (
        st.selectbox(
            "Select prediction to review",
            options=list(
                image_lookup.keys()
            ),
            key=(
                "review_image_"
                + context.category_key
            ),
        )
    )

    record = (
        image_lookup[
            selected_image_path
        ]
    )

    confidence = (
        record.get(
            "highest_confidence"
        )
    )

    info_columns = (
        st.columns(
            3
        )
    )

    info_columns[0].metric(
        "Review status",
        record.get(
            "review_status",
            "pending",
        ),
    )

    info_columns[1].metric(
        "Predicted boxes",
        len(
            record.get(
                "annotations",
                [],
            )
        ),
    )

    info_columns[2].metric(
        "Highest confidence",
        (
            f"{float(confidence):.3f}"
            if confidence is not None
            else "None"
        ),
    )

    st.markdown(
        "### AI prediction"
    )

    try:
        preview = (
            draw_annotations(
                report["root"],
                record,
            )
        )

        st.image(
            preview,
            caption=(
                "Orange box = AI prediction"
            ),
            width="stretch",
        )

    except Exception as exc:
        st.error(
            "Could not draw prediction: "
            f"{exc}"
        )

    prediction_rows = []

    for index, annotation in enumerate(
        record.get(
            "annotations",
            [],
        ),
        start=1,
    ):
        prediction_rows.append(
            {
                "box": index,
                "class_name": (
                    annotation.get(
                        "class_name"
                    )
                ),
                "confidence": (
                    annotation.get(
                        "confidence"
                    )
                ),
                "xmin": (
                    annotation.get(
                        "xmin"
                    )
                ),
                "ymin": (
                    annotation.get(
                        "ymin"
                    )
                ),
                "xmax": (
                    annotation.get(
                        "xmax"
                    )
                ),
                "ymax": (
                    annotation.get(
                        "ymax"
                    )
                ),
            }
        )

    if prediction_rows:
        st.dataframe(
            pd.DataFrame(
                prediction_rows
            ),
            width="stretch",
            hide_index=True,
        )

    else:
        st.warning(
            "AI produced no box. "
            "Draw one if the target "
            "object is present."
        )

    review_note = (
        st.text_input(
            "Review note",
            key=(
                "review_note_"
                + context.category_key
                + "_"
                + selected_image_path
            ),
        )
    )

    action_col1, action_col2, action_col3 = (
        st.columns(
            3
        )
    )

    verify_pressed = (
        action_col1.button(
            "Verify Prediction",
            type="primary",
            disabled=(
                not bool(
                    record.get(
                        "annotations"
                    )
                )
            ),
            key=(
                "verify_"
                + context.category_key
                + "_"
                + selected_image_path
            ),
        )
    )

    reject_pressed = (
        action_col2.button(
            "Reject Prediction",
            key=(
                "reject_"
                + context.category_key
                + "_"
                + selected_image_path
            ),
        )
    )

    no_object_pressed = (
        action_col3.button(
            "Mark No Object",
            key=(
                "no_object_"
                + context.category_key
                + "_"
                + selected_image_path
            ),
        )
    )

    try:
        if verify_pressed:
            manager.verify_prediction(
                selected_image_path,
                note=(
                    review_note
                    or None
                ),
            )

            st.success(
                "Prediction verified."
            )

            st.rerun(
                scope="fragment"
            )

        if reject_pressed:
            manager.reject_prediction(
                selected_image_path,
                note=(
                    review_note
                    or None
                ),
            )

            st.success(
                "Prediction rejected."
            )

            st.rerun(
                scope="fragment"
            )

        if no_object_pressed:
            manager.mark_no_object(
                selected_image_path,
                note=(
                    review_note
                    or None
                ),
            )

            st.success(
                "Image marked as no object."
            )

            st.rerun(
                scope="fragment"
            )

    except Exception as exc:
        st.error(
            "Review action failed: "
            f"{exc}"
        )

    # -----------------------------------------------------
    # DRAWABLE CORRECTION
    # -----------------------------------------------------

    with st.expander(
        "Draw or correct bounding boxes",
        expanded=(
            not bool(
                record.get(
                    "annotations"
                )
            )
        ),
    ):
        st.write(
            "Target object ke top-left "
            "corner se bottom-right corner "
            "tak mouse drag karo."
        )

        st.info(
            "Correction label: "
            f"{context.mapped_label}"
        )

        canvas_identity = (
            context.category_key
            + "|"
            + selected_image_path
        )

        safe_canvas_key = (
            "bbox_"
            + hashlib.sha256(
                canvas_identity.encode(
                    "utf-8"
                )
            ).hexdigest()[:20]
        )

        try:
            drawn_annotations = (
                show_drawable_bbox_editor(
                    dataset_root=(
                        report["root"]
                    ),
                    image_record=(
                        record
                    ),
                    class_name=(
                        context.mapped_label
                    ),
                    canvas_key=(
                        safe_canvas_key
                    ),
                    maximum_width=900,
                )
            )

            if st.button(
                "Save Drawn Correction",
                type="primary",
                disabled=(
                    not drawn_annotations
                ),
                key=(
                    "save_drawn_"
                    + safe_canvas_key
                ),
            ):
                manager.save_corrected_annotations(
                    image_path=(
                        selected_image_path
                    ),
                    annotations=(
                        drawn_annotations
                    ),
                    note=(
                        review_note
                        or
                        "Bounding boxes "
                        "drawn manually"
                    ),
                )

                st.success(
                    "Mouse-drawn correction saved."
                )

                st.rerun(
                    scope="fragment"
                )

        except DrawableEditorError as exc:
            st.error(
                "Bounding-box editor error: "
                f"{exc}"
            )

        except Exception as exc:
            st.error(
                "Could not save correction: "
                f"{exc}"
            )

    st.divider()

    if st.button(
        "Export Verified Review Annotations",
        key=(
            "export_verified_"
            + context.category_key
        ),
    ):
        try:
            verified_file = (
                paths[
                    "verified"
                ]
            )

            verified_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                manager.export_verified_annotations(
                verified_file
                )
            )

            st.success(
                "Verified annotations saved: "
                f"{output_path}"
            )

        except Exception as exc:
            st.error(
                f"Export failed: {exc}"
            )

    verified_file = (
        paths[
            "verified"
        ]
    )

    if verified_file.is_file():
        st.download_button(
            "Download Verified Annotation JSON",
            data=(
                verified_file
                .read_bytes()
            ),
            file_name=(
                context.category_key
                + "_verified_annotations.json"
            ),
            mime="application/json",
            key=(
                "download_verified_"
                + context.category_key
            ),
        )


# =========================================================
# UNIQUE ZIP TAB
# =========================================================

@st.fragment
def display_unique_zip_tab(
    report: dict,
):
    """
    Existing global Unique ZIP functionality.
    """

    (
        unique_records,
        excluded_records,
        corrupt_records,
        summary,
    ) = calculate_unique_image_summary(
        report
    )

    st.subheader(
        "Unique Images ZIP"
    )

    columns = st.columns(
        3
    )

    columns[0].metric(
        "Valid images",
        summary[
            "valid_images"
        ],
    )

    columns[1].metric(
        "Unique images",
        summary[
            "unique_images"
        ],
    )

    columns[2].metric(
        "Repeated files",
        summary[
            "repeated_files"
        ],
    )

    expected_path = (
        normalize_path(
            report["root"]
        )
        / "image_zip"
        / "unique_images.zip"
    )

    st.code(
        str(
            expected_path
        )
    )

    if st.button(
        "Create Unique Images ZIP",
        type="primary",
        key="create_zip_button",
    ):
        progress = (
            st.progress(
                0,
                text="Creating ZIP...",
            )
        )

        detail = st.empty()

        def update_progress(
            completed,
            total,
            relative_path,
        ):
            percentage = int(
                completed
                * 100
                / total
            )

            progress.progress(
                percentage,
                text=(
                    f"Adding "
                    f"{completed}/{total}"
                ),
            )

            detail.caption(
                relative_path
            )

        try:
            output_path = (
                create_unique_images_zip(
                    report,
                    progress_callback=(
                        update_progress
                    ),
                )
            )

            st.session_state[
                "unique_zip_path"
            ] = str(
                output_path
            )

            progress.progress(
                100,
                text="ZIP completed",
            )

            detail.empty()

            st.success(
                "Unique ZIP created."
            )

        except Exception as exc:
            progress.empty()

            detail.empty()

            st.error(
                "ZIP creation failed: "
                f"{exc}"
            )

    zip_value = (
        st.session_state.get(
            "unique_zip_path"
        )
    )

    if (
        not zip_value
        and expected_path.is_file()
    ):
        zip_value = str(
            expected_path
        )

    if zip_value:
        zip_path = Path(
            zip_value
        )

        if zip_path.is_file():
            st.success(
                "ZIP available: "
                f"{zip_path}"
            )

            st.download_button(
                "Download Unique Images ZIP",
                data=(
                    zip_path
                    .read_bytes()
                ),
                file_name=(
                    "unique_images.zip"
                ),
                mime=(
                    "application/zip"
                ),
                key=(
                    "download_unique_zip"
                ),
            )


# =========================================================
# DUPLICATE PREVIEW
# =========================================================


@st.fragment
def display_duplicate_tab(
    report: dict,
):
    """
    Show all exact SHA-256 duplicate groups together.

    A master-label file is not required.
    Images use compact thumbnails.
    """

    duplicates = report.get(
        "duplicates",
        {},
    )

    st.subheader(
        "Exact Duplicate Verification"
    )

    if not duplicates:
        st.success(
            "No exact duplicate groups detected."
        )

        return

    groups = list(
        duplicates.items()
    )

    repeated_file_count = sum(
        max(
            0,
            len(image_paths) - 1,
        )
        for _, image_paths
        in groups
    )

    summary_columns = st.columns(
        2
    )

    summary_columns[0].metric(
        "Duplicate Groups",
        len(
            groups
        ),
    )

    summary_columns[1].metric(
        "Repeated Files",
        repeated_file_count,
    )

    st.info(
        "All duplicate groups are shown below. "
        "Files inside the same group have the "
        "same SHA-256 hash."
    )

    record_lookup = {
        record[
            "relative_path"
        ]: record
        for record
        in report.get(
            "records",
            [],
        )
    }

    for group_number, (
        file_hash,
        image_paths,
    ) in enumerate(
        groups,
        start=1,
    ):
        st.markdown(
            f"### Duplicate Group {group_number}"
        )

        st.caption(
            f"{len(image_paths)} matching files | "
            f"{max(0, len(image_paths) - 1)} "
            "extra repeated copies"
        )

        with st.expander(
            "Show SHA-256",
            expanded=False,
        ):
            st.code(
                file_hash
            )

        columns_per_row = 4

        for start_index in range(
            0,
            len(image_paths),
            columns_per_row,
        ):
            current_paths = image_paths[
                start_index:
                start_index
                + columns_per_row
            ]

            columns = st.columns(
                len(
                    current_paths
                )
            )

            for column, relative_path in zip(
                columns,
                current_paths,
            ):
                record = record_lookup.get(
                    relative_path
                )

                with column:
                    if record is None:
                        st.warning(
                            "Image record not found."
                        )

                        st.caption(
                            relative_path
                        )

                        continue

                    image_path = Path(
                        record[
                            "path"
                        ]
                    )

                    if image_path.is_file():
                        st.image(
                            str(
                                image_path
                            ),
                            width="stretch",
                        )

                    else:
                        st.warning(
                            "Image file is missing."
                        )

                    st.caption(
                        relative_path
                    )

        st.divider()


# =========================================================
# INITIALIZATION
# =========================================================

config = load_config()

initialize_state(
    config
)

mapping_path = get_mapping_path(
    config
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header(
        "Dataset Configuration"
    )

    st.text_input(
        "Dataset root directory",
        key="dataset_root_value",
    )

    st.caption(
        "Duplicate scan/delete ke liye sirf folder "
        "path required hai. Label file ki zarurat "
        "nahi hai."
    )

    if st.button(
        "Scan Dataset",
        type="primary",
        key="scan_dataset",
    ):
        try:
            with st.spinner(
                "Scanning dataset..."
            ):
                image_count = (
                    perform_scan(
                        config,
                        mapping_path,
                    )
                )

            st.success(
                f"Scanned "
                f"{image_count} images."
            )

        except Exception as exc:
            st.session_state.report = (
                None
            )

            st.error(
                "Dataset scan failed: "
                f"{exc}"
            )


# =========================================================
# MAIN UI
# =========================================================

report = st.session_state.get(
    "report"
)

if report is None:
    st.info(
        "Sidebar me Dataset Root set karo, "
        "phir Scan Dataset select karo. "
        "Duplicate-only scan ke liye "
        "master-label file optional hai."
    )

else:
    (
        unique_records,
        excluded_records,
        corrupt_records,
        dataset_summary,
    ) = calculate_unique_image_summary(
        report
    )

    summary_columns = (
        st.columns(
            5
        )
    )

    summary_columns[0].metric(
        "Images",
        dataset_summary[
            "total_images"
        ],
    )

    summary_columns[1].metric(
        "Unique",
        dataset_summary[
            "unique_images"
        ],
    )

    summary_columns[2].metric(
        "Repeated",
        dataset_summary[
            "repeated_files"
        ],
    )

    summary_columns[3].metric(
        "Duplicate Groups",
        dataset_summary[
            "duplicate_groups"
        ],
    )

    summary_columns[4].metric(
        "Corrupted",
        dataset_summary[
            "corrupt_images"
        ],
    )

    # =====================================================
    # DYNAMIC WORKING FOLDER
    # =====================================================

    discovered_folders = (
        discover_image_folders(
            report["root"]
        )
    )

    current_mapping = (
        normalize_mapping(
            st.session_state.mapping
        )
    )

    st.subheader(
        "Working Folder"
    )

    selected_folder = None
    selected_label = ""
    context = None
    dynamic_paths = None

    if not discovered_folders:
        st.info(
            "Images selected Dataset Root me directly "
            "present hain. Duplicate Preview, Unique "
            "Images ZIP, Scan Report aur corrupt-image "
            "checking available hain. AI automation ke "
            "liye parent dataset root, image folder and "
            "valid label mapping required hogi."
        )

    else:
        current_folder = (
            st.session_state.get(
                "selected_working_folder"
            )
        )

        if (
            current_folder
            in discovered_folders
        ):
            default_index = (
                discovered_folders.index(
                    current_folder
                )
            )

        else:
            default_index = 0

        selected_folder = (
            st.selectbox(
                "Select folder/subfolder "
                "for automation",
                options=(
                    discovered_folders
                ),
                index=(
                    default_index
                ),
                key=(
                    "working_folder_selector"
                ),
            )
        )

        st.session_state[
            "selected_working_folder"
        ] = selected_folder

        selected_label = (
            current_mapping.get(
                selected_folder,
                "",
            )
        )

        (
            selected_info_col1,
            selected_info_col2,
        ) = st.columns(
            2
        )

        selected_info_col1.info(
            "Selected folder: "
            f"{selected_folder}"
        )

        if selected_label:
            selected_info_col2.success(
                "Mapped label: "
                f"{selected_label}"
            )

        else:
            selected_info_col2.warning(
                "No mapped label. Duplicate Preview, "
                "Unique ZIP and Scan Report still work. "
                "AI automation requires a valid mapping."
            )

        if selected_label:
            try:
                context = (
                    CategoryContext.create(
                        project_root=(
                            PROJECT_ROOT
                        ),
                        dataset_root=(
                            report["root"]
                        ),
                        selected_folder=(
                            selected_folder
                        ),
                        mapped_label=(
                            selected_label
                        ),
                    )
                )

                context.ensure_working_directories()

                dynamic_paths = (
                    legacy_or_dynamic_paths(
                        context
                    )
                )

            except CategoryContextError as exc:
                st.error(
                    str(
                        exc
                    )
                )


    # =====================================================
    # TABS
    # =====================================================

    (
        dataset_tab,
        annotations_tab,
        review_tab,
        utilities_tab,
    ) = st.tabs(
        [
            "Dataset",
            "Annotations",
            "AI Review",
            "Utilities",
        ]
    )

    # =====================================================
    # DATASET TAB
    # =====================================================

    with dataset_tab:
        st.subheader(
            "Folder-to-Label Mapping"
        )

        labels = (
            st.session_state.labels
        )

        if not labels:
            st.info(
                "Master label file load nahi hui. "
                "Duplicate detection, all-groups "
                "Duplicate Preview, Unique Images ZIP, "
                "Scan Report aur corrupt-image checking "
                "available hain. Folder mapping, "
                "annotation import, training and AI "
                "automation ke liye valid master label "
                "file required hogi."
            )

        elif not discovered_folders:
            st.info(
                "Selected Dataset Root me images "
                "directly present hain. Duplicate tools "
                "available hain. Folder-to-label mapping "
                "aur AI automation ke liye parent "
                "dataset root scan karo."
            )

        else:
            with st.form(
                "folder_mapping_form",
                clear_on_submit=False,
            ):
                edited_mapping = {}

                for folder in discovered_folders:
                    widget_key = (
                        "folder_label_"
                        + folder.replace(
                            "/",
                            "__",
                        )
                        .replace(
                            "\\",
                            "__",
                        )
                    )

                    existing_label = (
                        current_mapping.get(
                            folder,
                            "",
                        )
                    )

                    if (
                        widget_key
                        not in st.session_state
                    ):
                        if (
                            existing_label
                            in labels
                        ):
                            st.session_state[
                                widget_key
                            ] = existing_label

                        else:
                            st.session_state[
                                widget_key
                            ] = ""

                    if (
                        st.session_state[
                            widget_key
                        ]
                        not in labels
                    ):
                        st.session_state[
                            widget_key
                        ] = ""

                    image_count = (
                        report.get(
                            "folder_counts",
                            {},
                        ).get(
                            folder,
                            "",
                        )
                    )

                    label_text = (
                        f"{folder}"
                    )

                    if image_count != "":
                        label_text += (
                            f" ({image_count} images)"
                        )

                    edited_mapping[
                        folder
                    ] = st.selectbox(
                        label_text,
                        options=[
                            "",
                            *labels,
                        ],
                        key=(
                            widget_key
                        ),
                    )

                save_mapping = (
                    st.form_submit_button(
                        "Validate and Save "
                        "Folder Mapping",
                        type="primary",
                    )
                )

            if save_mapping:
                try:
                    clean_mapping = {
                        folder: label
                        for folder, label
                        in edited_mapping.items()
                        if label
                    }

                    dataset_root = normalize_path(
                        report["root"]
                    )

                    label_file = (
                        dataset_root
                        / st.session_state[
                            "label_filename_value"
                        ]
                    )

                    registry = LabelRegistry(
                        label_file,
                        mapping_path,
                    ).load()

                    registry.save_mapping(
                        clean_mapping
                    )

                    st.session_state[
                        "mapping"
                    ] = normalize_mapping(
                        clean_mapping
                    )

                    st.success(
                        "Folder mappings saved. "
                        "Select a working folder above."
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        "Mapping failed: "
                        f"{exc}"
                    )

        st.subheader(
            "Image Validation"
        )

        show_table = (
            st.checkbox(
                "Show full "
                "image-validation table",
                value=False,
                key=(
                    "show_validation_table"
                ),
            )
        )

        if show_table:
            st.dataframe(
                pd.DataFrame(
                    report.get(
                        "records",
                        [],
                    )
                ),
                width="stretch",
                hide_index=True,
            )

        if corrupt_records:
            with st.expander(
                "Corrupted Images"
            ):
                st.dataframe(
                    pd.DataFrame(
                        corrupt_records
                    ),
                    width="stretch",
                    hide_index=True,
                )


    # =====================================================
    # ANNOTATIONS TAB
    # =====================================================

    with annotations_tab:
        if (
            context is not None
            and dynamic_paths is not None
        ):
            display_annotation_import_tab(
                report=report,
                context=context,
                paths=(
                    dynamic_paths
                ),
            )

        else:
            st.warning(
                "Selected folder needs a "
                "valid label mapping first."
            )

    # =====================================================
    # AI REVIEW TAB
    # =====================================================

    with review_tab:
        if (
            context is not None
            and dynamic_paths is not None
        ):
            display_ai_review_tab(
                report=report,
                context=context,
                paths=(
                    dynamic_paths
                ),
            )

        else:
            st.warning(
                "Selected folder needs a "
                "valid label mapping first."
            )

    # =====================================================
    # UTILITIES TAB
    # =====================================================

    with utilities_tab:
        utility_choice = st.radio(
            "Select utility",
            options=[
                "Delete Duplicates",
                "Rename Images",
                "Collect Images",
                "Find Similar Images",
                "Scan Report",
            ],
            horizontal=True,
            key="utility_choice",
        )

        def rescan_utility_folder():
            perform_scan(
                config,
                mapping_path,
            )

        if (
            utility_choice
            == "Delete Duplicates"
        ):
            display_duplicate_management(
                report=report,
                rescan_callback=(
                    rescan_utility_folder
                ),
            )

        elif (
            utility_choice
            == "Rename Images"
        ):
            display_image_rename(
                report=report,
                rescan_callback=(
                    rescan_utility_folder
                ),
            )

        elif (
            utility_choice
            == "Collect Images"
        ):
            display_image_collector()

        elif (
            utility_choice
            == "Find Similar Images"
        ):
            display_similar_image_search()

        else:
            st.subheader(
                "Dataset Scan Report"
            )

            st.download_button(
                "Download Scan Report JSON",
                data=json.dumps(
                    report,
                    indent=2,
                ),
                file_name=(
                    "dataset_scan_report.json"
                ),
                mime="application/json",
                key="download_scan_report",
            )

