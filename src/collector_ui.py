from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.image_collector import (
    ImageCollectorError,
    build_collection_plan,
    collect_classified_images,
)


def _safe_key(
    prefix: str,
    value: str,
) -> str:
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]

    return f"{prefix}_{digest}"


@st.dialog(
    "Confirm Image Collection"
)
def confirm_image_collection(
    source_folder: str,
    output_folder: str,
) -> None:

    st.warning(
        "Images will be COPIED to the output folder. "
        "Original source images will not be deleted "
        "or moved."
    )

    st.write(
        f"Source: **{source_folder}**"
    )

    st.write(
        f"Output: **{output_folder}**"
    )

    confirmed = st.checkbox(
        "I want to copy these classified images.",
        key=_safe_key(
            "collector_confirm",
            source_folder
            + "|"
            + output_folder,
        ),
    )

    left, right = st.columns(2)

    with left:
        copy_pressed = st.button(
            "Yes, Collect Images",
            type="primary",
            disabled=not confirmed,
            width="stretch",
            key=_safe_key(
                "collector_yes",
                source_folder
                + "|"
                + output_folder,
            ),
        )

    with right:
        cancel_pressed = st.button(
            "Cancel",
            width="stretch",
            key=_safe_key(
                "collector_cancel",
                source_folder
                + "|"
                + output_folder,
            ),
        )

    if copy_pressed:
        try:
            result = collect_classified_images(
                source_folder=source_folder,
                output_folder=output_folder,
            )

            st.session_state[
                "collector_result"
            ] = result

            st.rerun()

        except ImageCollectorError as exc:
            st.error(
                f"Collection failed: {exc}"
            )

        except Exception as exc:
            st.error(
                f"Unexpected collection error: {exc}"
            )

    if cancel_pressed:
        st.rerun()


@st.fragment
def display_image_collector() -> None:

    st.subheader(
        "Collect Classified Images"
    )

    st.caption(
        "Multiple site folders se categorized images "
        "ek common output structure me copy karo."
    )

    source_folder = st.text_input(
        "Source folder path",
        key="collector_source_folder",
        placeholder=(
            r"C:\Users\304758\Downloads\Nokia Sites"
        ),
    )

    output_folder = st.text_input(
        "Output folder path",
        key="collector_output_folder",
        placeholder=(
            r"C:\Users\304758\Downloads\Collected Images"
        ),
    )

    previous_result = st.session_state.get(
        "collector_result"
    )

    if isinstance(
        previous_result,
        dict,
    ):
        st.success(
            "Image collection completed. "
            f"Copied: "
            f"{previous_result.get('images_copied', 0)} | "
            f"Failed: "
            f"{previous_result.get('images_failed', 0)}"
        )

        st.write(
            "Output folder:"
        )

        st.code(
            previous_result.get(
                "output_folder",
                "",
            )
        )

        del st.session_state[
            "collector_result"
        ]

    preview_pressed = st.button(
        "Preview Collection Plan",
        key="preview_collection_plan",
    )

    if not preview_pressed:
        return

    source_folder = source_folder.strip()
    output_folder = output_folder.strip()

    if not source_folder:
        st.error(
            "Source folder path required hai."
        )
        return

    if not output_folder:
        st.error(
            "Output folder path required hai."
        )
        return

    try:
        plan = build_collection_plan(
            source_folder=source_folder,
            output_folder=output_folder,
        )

    except ImageCollectorError as exc:
        st.error(
            f"Preview failed: {exc}"
        )
        return

    except Exception as exc:
        st.error(
            f"Unexpected preview error: {exc}"
        )
        return

    columns = st.columns(3)

    columns[0].metric(
        "Images Found",
        plan[
            "image_count"
        ],
    )

    columns[1].metric(
        "Categories",
        plan[
            "category_count"
        ],
    )

    columns[2].metric(
        "Skipped",
        plan[
            "skipped_without_category_count"
        ],
    )

    st.markdown(
        "### Categories found"
    )

    for category in plan[
        "categories"
    ]:
        st.write(
            f"- {category}"
        )

    preview_rows = []

    for record in plan[
        "records"
    ][:100]:

        preview_rows.append(
            {
                "Source": (
                    record[
                        "source_relative"
                    ]
                ),
                "Category": (
                    record[
                        "category_path"
                    ]
                ),
                "Image": (
                    record[
                        "image_name"
                    ]
                ),
                "Output": (
                    record[
                        "proposed_destination"
                    ]
                ),
            }
        )

    if preview_rows:
        st.dataframe(
            pd.DataFrame(
                preview_rows
            ),
            width="stretch",
            hide_index=True,
        )

    if plan[
        "image_count"
    ] > 100:
        st.caption(
            "First 100 planned copies shown. "
            f"Total images: "
            f"{plan['image_count']}"
        )

    st.info(
        "Preview only. Original files have not "
        "been changed."
    )

    if st.button(
        "Collect Images",
        type="primary",
        key=_safe_key(
            "open_collection_confirmation",
            source_folder
            + "|"
            + output_folder,
        ),
    ):
        confirm_image_collection(
            source_folder=source_folder,
            output_folder=output_folder,
        )
