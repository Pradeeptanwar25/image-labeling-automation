from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.image_renamer import (
    ImageRenameError,
    preview_rename_plan,
    rename_images_with_folder_prefix,
)


def _root_key(report: dict[str, Any]) -> str:
    root = str(
        report.get(
            "root",
            "",
        )
    )

    return hashlib.sha256(
        root.encode("utf-8")
    ).hexdigest()[:16]


@st.dialog("Confirm Image Rename")
def confirm_image_rename(
    report: dict[str, Any],
    rescan_callback: Callable[[], None],
) -> None:
    folder = Path(
        str(report["root"])
    ).resolve()

    plan = preview_rename_plan(
        folder
    )

    st.warning(
        "Images will be renamed directly inside "
        "the scanned source folder."
    )

    st.write(
        f"Folder: **{folder}**"
    )

    st.write(
        f"Folder prefix: **{plan['folder_prefix']}**"
    )

    st.write(
        f"Images to rename: "
        f"**{plan['files_to_rename']}**"
    )

    confirmed = st.checkbox(
        "I understand that the image filenames "
        "will be changed.",
        key=(
            "rename_confirm_"
            + _root_key(report)
        ),
    )

    left, right = st.columns(2)

    with left:
        rename_pressed = st.button(
            "Yes, Rename Images",
            type="primary",
            disabled=not confirmed,
            width="stretch",
            key=(
                "rename_yes_"
                + _root_key(report)
            ),
        )

    with right:
        cancel_pressed = st.button(
            "Cancel",
            width="stretch",
            key=(
                "rename_cancel_"
                + _root_key(report)
            ),
        )

    if rename_pressed:
        try:
            result = (
                rename_images_with_folder_prefix(
                    folder
                )
            )

            st.session_state[
                "image_rename_result"
            ] = result

            rescan_callback()

            st.rerun()

        except ImageRenameError as exc:
            st.error(
                f"Rename failed: {exc}"
            )

        except Exception as exc:
            st.error(
                f"Unexpected rename error: {exc}"
            )

    if cancel_pressed:
        st.rerun()


@st.fragment
def display_image_rename(
    report: dict[str, Any],
    rescan_callback: Callable[[], None],
) -> None:
    folder = Path(
        str(report["root"])
    ).resolve()

    st.subheader(
        "Rename Images"
    )

    st.caption(
        f"Scanned folder: {folder}"
    )

    try:
        plan = preview_rename_plan(
            folder
        )

    except ImageRenameError as exc:
        st.warning(
            str(exc)
        )

        return

    columns = st.columns(3)

    columns[0].metric(
        "Images",
        plan[
            "total_images"
        ],
    )

    columns[1].metric(
        "Need Rename",
        plan[
            "files_to_rename"
        ],
    )

    columns[2].metric(
        "Prefix",
        plan[
            "folder_prefix"
        ],
    )

    previous_result = st.session_state.get(
        "image_rename_result"
    )

    if isinstance(
        previous_result,
        dict,
    ):
        st.success(
            "Image rename completed. "
            f"Renamed: "
            f"{previous_result.get('renamed_count', 0)}"
        )

        del st.session_state[
            "image_rename_result"
        ]

    if plan[
        "files_to_rename"
    ] == 0:
        st.success(
            "All images already have the expected "
            "sequential folder-based names."
        )

        return

    st.info(
        "Preview only. No filename changes happen "
        "until Rename Images is confirmed."
    )

    changed_rows = [
        item
        for item
        in plan[
            "renamed"
        ]
        if not item[
            "unchanged"
        ]
    ]

    preview_rows = changed_rows[
        :100
    ]

    st.dataframe(
        pd.DataFrame(
            preview_rows
        ),
        width="stretch",
        hide_index=True,
    )

    if len(
        changed_rows
    ) > 100:
        st.caption(
            f"Showing first 100 of "
            f"{len(changed_rows)} rename operations."
        )

    st.warning(
        "The rename happens directly in the scanned "
        "folder. Existing image content is not changed."
    )

    if st.button(
        "Rename Images",
        type="primary",
        key=(
            "open_rename_dialog_"
            + _root_key(report)
        ),
    ):
        confirm_image_rename(
            report=report,
            rescan_callback=(
                rescan_callback
            ),
        )
