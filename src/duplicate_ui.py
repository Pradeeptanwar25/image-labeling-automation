from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.duplicate_manager import (
    DuplicateManagerError,
    delete_root_duplicates,
    get_root_duplicate_groups,
    get_root_duplicate_summary,
)


def _root_key(
    report: dict[str, Any],
) -> str:

    root = str(
        report.get(
            "root",
            "",
        )
    )

    return hashlib.sha256(
        root.encode("utf-8")
    ).hexdigest()[:16]


@st.dialog(
    "Confirm Duplicate Deletion"
)
def confirm_duplicate_deletion(
    report: dict[str, Any],
    rescan_callback: Callable[[], None],
) -> None:

    summary = get_root_duplicate_summary(
        report
    )

    dataset_root = str(
        report["root"]
    )

    st.error(
        "This permanently deletes extra exact "
        "duplicate files from the scanned folder."
    )

    st.write(
        f"Folder: **{dataset_root}**"
    )

    st.write(
        "Duplicate groups: "
        f"**{summary['duplicate_groups']}**"
    )

    st.write(
        "Files to delete: "
        f"**{summary['files_to_delete']}**"
    )

    st.info(
        "One file from every SHA-256 duplicate "
        "group will always be retained."
    )

    key_hash = _root_key(
        report
    )

    confirmed = st.checkbox(
        "I understand these duplicate copies "
        "will be permanently deleted.",
        key=(
            "confirm_duplicate_"
            + key_hash
        ),
    )

    left, right = st.columns(
        2
    )

    with left:
        delete_pressed = st.button(
            "Yes, Delete Duplicates",
            type="primary",
            disabled=not confirmed,
            width="stretch",
            key=(
                "delete_duplicates_"
                + key_hash
            ),
        )

    with right:
        cancel_pressed = st.button(
            "Cancel",
            width="stretch",
            key=(
                "cancel_duplicates_"
                + key_hash
            ),
        )

    if delete_pressed:
        try:
            result = delete_root_duplicates(
                report
            )

            st.session_state[
                "duplicate_delete_result"
            ] = result

            rescan_callback()

            st.rerun()

        except DuplicateManagerError as exc:
            st.error(
                "Deletion failed: "
                f"{exc}"
            )

        except Exception as exc:
            st.error(
                "Unexpected deletion error: "
                f"{exc}"
            )

    if cancel_pressed:
        st.rerun()


@st.fragment
def display_duplicate_management(
    report: dict[str, Any],
    rescan_callback: Callable[[], None],
) -> None:

    st.subheader(
        "Duplicate Image Management"
    )

    dataset_root = Path(
        str(
            report["root"]
        )
    ).resolve()

    groups = get_root_duplicate_groups(
        report
    )

    summary = get_root_duplicate_summary(
        report
    )

    st.caption(
        f"Scanned folder: {dataset_root}"
    )

    columns = st.columns(
        3
    )

    columns[0].metric(
        "Duplicate Groups",
        summary[
            "duplicate_groups"
        ],
    )

    columns[1].metric(
        "Extra Copies",
        summary[
            "files_to_delete"
        ],
    )

    columns[2].metric(
        "Copies Retained",
        summary[
            "files_to_retain"
        ],
    )

    previous_result = st.session_state.get(
        "duplicate_delete_result"
    )

    if isinstance(
        previous_result,
        dict,
    ):
        st.success(
            "Duplicate deletion completed. "
            f"Deleted: "
            f"{previous_result.get('deleted_count', 0)}, "
            f"failed: "
            f"{previous_result.get('failed_count', 0)}."
        )

        failures = previous_result.get(
            "failed_files",
            [],
        )

        if failures:
            st.dataframe(
                pd.DataFrame(
                    failures
                ),
                width="stretch",
                hide_index=True,
            )

        del st.session_state[
            "duplicate_delete_result"
        ]

    if not groups:
        st.success(
            "No exact duplicate images found."
        )

        return

    st.info(
        "Green = file that will remain. "
        "Red = extra exact copy that will "
        "be deleted after confirmation."
    )

    for group_number, group in enumerate(
        groups,
        start=1,
    ):

        retained = group[
            "retained"
        ]

        duplicates = group[
            "duplicates"
        ]

        with st.expander(
            (
                f"Duplicate Group "
                f"{group_number} - "
                f"{len(duplicates)} extra copy/copies"
            ),
            expanded=(
                group_number <= 3
            ),
        ):

            st.caption(
                f"SHA-256: "
                f"{group['sha256']}"
            )

            st.markdown(
                "**KEEP**"
            )

            retained_path = (
                dataset_root
                / Path(
                    retained[
                        "relative_path"
                    ]
                )
            )

            keep_image, keep_text = (
                st.columns(
                    [
                        1,
                        3,
                    ]
                )
            )

            with keep_image:
                if retained_path.is_file():
                    st.image(
                        str(
                            retained_path
                        ),
                        width=140,
                    )

            with keep_text:
                st.success(
                    retained[
                        "relative_path"
                    ]
                )

                st.caption(
                    "This file will remain."
                )

            st.markdown(
                "**DELETE**"
            )

            for start_index in range(
                0,
                len(
                    duplicates
                ),
                4,
            ):

                batch = duplicates[
                    start_index:
                    start_index + 4
                ]

                preview_columns = st.columns(
                    len(
                        batch
                    )
                )

                for column, record in zip(
                    preview_columns,
                    batch,
                ):

                    file_path = (
                        dataset_root
                        / Path(
                            record[
                                "relative_path"
                            ]
                        )
                    )

                    with column:
                        if file_path.is_file():
                            st.image(
                                str(
                                    file_path
                                ),
                                width=140,
                            )

                        st.caption(
                            record[
                                "relative_path"
                            ]
                        )

                        st.error(
                            "Will be deleted"
                        )

    st.warning(
        f"{summary['files_to_delete']} exact "
        "duplicate file(s) will be permanently "
        "deleted after confirmation."
    )

    if st.button(
        "Delete Duplicate Files",
        type="primary",
        key=(
            "open_duplicate_confirmation_"
            + _root_key(
                report
            )
        ),
    ):

        confirm_duplicate_deletion(
            report=report,
            rescan_callback=(
                rescan_callback
            ),
        )
