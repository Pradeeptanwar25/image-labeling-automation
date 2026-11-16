from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from streamlit_drawable_canvas import st_canvas


class DrawableEditorError(Exception):
    """
    Raised when the interactive bounding-box editor
    cannot process an image.
    """

    pass


def _calculate_display_size(
    image_width: int,
    image_height: int,
    maximum_width: int = 900,
):
    """
    Resize image for browser display while
    preserving aspect ratio.
    """

    if image_width <= 0:
        raise DrawableEditorError(
            "Invalid image width."
        )

    if image_height <= 0:
        raise DrawableEditorError(
            "Invalid image height."
        )

    display_width = min(
        image_width,
        maximum_width,
    )

    scale = (
        display_width
        / image_width
    )

    display_height = round(
        image_height
        * scale
    )

    display_width = max(
        1,
        int(display_width),
    )

    display_height = max(
        1,
        int(display_height),
    )

    return (
        display_width,
        display_height,
    )


def _validate_box(
    box,
    image_width,
    image_height,
    class_name,
):
    """
    Validate a box returned by the drawable canvas.

    The latest drawable-canvas component returns
    boxes in original background-image coordinates
    through boxes_in_image_space.
    """

    try:
        left = float(
            box.get(
                "left",
                0.0,
            )
        )

        top = float(
            box.get(
                "top",
                0.0,
            )
        )

        width = float(
            box.get(
                "width",
                0.0,
            )
        )

        height = float(
            box.get(
                "height",
                0.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if width <= 0:
        return None

    if height <= 0:
        return None

    xmin = max(
        0.0,
        left,
    )

    ymin = max(
        0.0,
        top,
    )

    xmax = min(
        float(image_width),
        left + width,
    )

    ymax = min(
        float(image_height),
        top + height,
    )

    if xmax <= xmin:
        return None

    if ymax <= ymin:
        return None

    box_label = str(
        box.get(
            "label",
            class_name,
        )
        or class_name
    ).strip()

    return {
        "class_name": box_label,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def show_drawable_bbox_editor(
    dataset_root: str | Path,
    image_record: dict[str, Any],
    class_name: str,
    canvas_key: str,
    maximum_width: int = 900,
):
    """
    Show interactive bounding-box editor.

    User can draw one or more rectangles directly
    over the original image.

    Returned coordinates are converted into the
    original source-image coordinate system.

    Original image is never modified.
    """

    dataset_root = Path(
        dataset_root
    ).resolve()

    relative_image_path = Path(
        image_record[
            "image_path"
        ]
    )

    source_image_path = (
        dataset_root
        / relative_image_path
    )

    if not source_image_path.is_file():
        raise DrawableEditorError(
            "Original image not found: "
            f"{source_image_path}"
        )

    try:
        with Image.open(
            source_image_path
        ) as source_image:

            original_image = (
                source_image
                .convert("RGB")
                .copy()
            )

    except Exception as exc:
        raise DrawableEditorError(
            "Could not open image: "
            f"{exc}"
        ) from exc

    (
        original_width,
        original_height,
    ) = original_image.size

    (
        display_width,
        display_height,
    ) = _calculate_display_size(
        image_width=original_width,
        image_height=original_height,
        maximum_width=maximum_width,
    )

    display_image = (
        original_image.resize(
            (
                display_width,
                display_height,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    st.info(
        "Target object ke top-left se bottom-right "
        "tak mouse drag karke green rectangle banao."
    )

    # -----------------------------------------------------
    # Drawable canvas
    # -----------------------------------------------------

    canvas_result = st_canvas(
        fill_color=(
            "rgba(0, 255, 0, 0.18)"
        ),
        stroke_width=3,
        stroke_color="#00ff00",
        background_image=display_image,
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="labeled_rect",
        label=class_name,
        return_image_data=True,
        key=canvas_key,
    )

    corrected_annotations = []

    # -----------------------------------------------------
    # Preferred API:
    #
    # boxes_in_image_space already contains coordinates
    # converted from canvas pixels back into the original
    # background-image coordinate system.
    # -----------------------------------------------------

    boxes_in_image_space = getattr(
        canvas_result,
        "boxes_in_image_space",
        None,
    )

    if boxes_in_image_space:

        for box in boxes_in_image_space:

            annotation = _validate_box(
                box=box,
                image_width=original_width,
                image_height=original_height,
                class_name=class_name,
            )

            if annotation is not None:

                corrected_annotations.append(
                    annotation
                )

    # -----------------------------------------------------
    # Compatibility fallback
    #
    # If boxes_in_image_space is unavailable but the
    # component exposes box data in canvas coordinates,
    # convert those coordinates manually.
    # -----------------------------------------------------

    elif getattr(
        canvas_result,
        "boxes",
        None,
    ):

        scale_x = (
            original_width
            / display_width
        )

        scale_y = (
            original_height
            / display_height
        )

        for box in canvas_result.boxes:

            try:

                canvas_left = float(
                    box.get(
                        "left",
                        0.0,
                    )
                )

                canvas_top = float(
                    box.get(
                        "top",
                        0.0,
                    )
                )

                canvas_width = float(
                    box.get(
                        "width",
                        0.0,
                    )
                )

                canvas_height = float(
                    box.get(
                        "height",
                        0.0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            converted_box = {
                "label": box.get(
                    "label",
                    class_name,
                ),
                "left": (
                    canvas_left
                    * scale_x
                ),
                "top": (
                    canvas_top
                    * scale_y
                ),
                "width": (
                    canvas_width
                    * scale_x
                ),
                "height": (
                    canvas_height
                    * scale_y
                ),
            }

            annotation = _validate_box(
                box=converted_box,
                image_width=original_width,
                image_height=original_height,
                class_name=class_name,
            )

            if annotation is not None:

                corrected_annotations.append(
                    annotation
                )

    # -----------------------------------------------------
    # UI status
    # -----------------------------------------------------

    if corrected_annotations:

        st.success(
            f"{len(corrected_annotations)} "
            "rectangle ready to save."
        )

        for index, annotation in enumerate(
            corrected_annotations,
            start=1,
        ):

            st.write(
                f"**Box {index}**"
            )

            st.code(
                (
                    f"Label: "
                    f"{annotation['class_name']}\n"
                    f"xmin: "
                    f"{annotation['xmin']:.1f}\n"
                    f"ymin: "
                    f"{annotation['ymin']:.1f}\n"
                    f"xmax: "
                    f"{annotation['xmax']:.1f}\n"
                    f"ymax: "
                    f"{annotation['ymax']:.1f}"
                )
            )

    else:

        st.warning(
            "Abhi rectangle draw nahi hua hai."
        )

    return corrected_annotations