from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import streamlit as st

from src.similar_image_finder import (
    SUPPORTED_EXTENSIONS,
    SimilarImageFinder,
    SimilarImageFinderError,
    copy_similarity_matches,
)


@st.cache_resource(
    show_spinner=False
)
def get_similarity_model():
    """
    Load DINOv2 once and reuse it during
    the Streamlit session.
    """

    return SimilarImageFinder(
        model_name="facebook/dinov2-small"
    )


def parse_reference_image_paths(
    raw_value: str,
) -> list[str]:
    """
    Parse reference-image paths.

    Recommended input:
        One image path per line.

    Also supported:
        Comma-separated quoted or unquoted paths.

    Duplicate paths are removed while preserving
    their original order.
    """

    clean_value = str(
        raw_value or ""
    ).strip()

    if not clean_value:
        return []

    raw_paths: list[str] = []

    for line in clean_value.splitlines():
        clean_line = line.strip()

        if not clean_line:
            continue

        try:
            parsed_values = next(
                csv.reader(
                    [clean_line],
                    skipinitialspace=True,
                )
            )
        except csv.Error:
            parsed_values = [
                clean_line
            ]

        for parsed_value in parsed_values:
            clean_path = (
                str(parsed_value)
                .strip()
                .strip('"')
                .strip("'")
            )

            if clean_path:
                raw_paths.append(
                    clean_path
                )

    unique_paths: list[str] = []
    seen_paths: set[str] = set()

    for raw_path in raw_paths:
        normalized_key = (
            str(
                Path(raw_path)
                .expanduser()
            )
            .casefold()
        )

        if normalized_key in seen_paths:
            continue

        seen_paths.add(
            normalized_key
        )

        unique_paths.append(
            raw_path
        )

    return unique_paths


def validate_reference_paths(
    raw_paths: list[str],
) -> tuple[list[Path], list[dict[str, str]]]:
    """
    Validate every reference path without starting
    the DINOv2 model.
    """

    valid_paths: list[Path] = []
    errors: list[dict[str, str]] = []

    for raw_path in raw_paths:
        try:
            path = (
                Path(raw_path)
                .expanduser()
                .resolve()
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            errors.append(
                {
                    "path": raw_path,
                    "error": str(exc),
                }
            )
            continue

        if not path.is_file():
            errors.append(
                {
                    "path": str(path),
                    "error": (
                        "File not found."
                    ),
                }
            )
            continue

        if (
            path.suffix.casefold()
            not in SUPPORTED_EXTENSIONS
        ):
            errors.append(
                {
                    "path": str(path),
                    "error": (
                        "Unsupported image type."
                    ),
                }
            )
            continue

        valid_paths.append(
            path
        )

    return valid_paths, errors


def display_reference_previews(
    reference_paths: list[Path],
) -> None:
    """
    Display compact previews for all valid references.
    """

    if not reference_paths:
        return

    st.markdown(
        "### Reference Images"
    )

    st.caption(
        f"{len(reference_paths)} valid reference "
        "image(s) selected."
    )

    columns_per_row = 4

    for start_index in range(
        0,
        len(reference_paths),
        columns_per_row,
    ):
        current_paths = reference_paths[
            start_index:
            start_index + columns_per_row
        ]

        columns = st.columns(
            len(current_paths)
        )

        for column, reference_path in zip(
            columns,
            current_paths,
        ):
            with column:
                st.image(
                    str(reference_path),
                    width=180,
                )

                st.caption(
                    reference_path.name
                )

                with st.expander(
                    "Full path",
                    expanded=False,
                ):
                    st.code(
                        str(reference_path)
                    )


def display_match_previews(
    matches: list[dict],
) -> None:
    """
    Display matched images in a compact grid.
    """

    if not matches:
        return

    st.markdown(
        "### Similar Image Preview"
    )

    columns_per_row = 4

    for start_index in range(
        0,
        len(matches),
        columns_per_row,
    ):
        current_matches = matches[
            start_index:
            start_index + columns_per_row
        ]

        columns = st.columns(
            len(current_matches)
        )

        for column, match in zip(
            columns,
            current_matches,
        ):
            with column:
                image_path = Path(
                    match.get(
                        "image_path",
                        "",
                    )
                )

                if image_path.is_file():
                    st.image(
                        str(image_path),
                        width="stretch",
                    )
                else:
                    st.warning(
                        "Image file is missing."
                    )

                st.markdown(
                    "**Similarity: "
                    f"{match.get('similarity_percent', 0):.2f}%**"
                )

                st.caption(
                    match.get(
                        "relative_path",
                        image_path.name,
                    )
                )

                references_agreed = int(
                    match.get(
                        "references_agreed",
                        0,
                    )
                )

                reference_count = int(
                    match.get(
                        "reference_count",
                        0,
                    )
                )

                best_positive_percent = match.get(
                    "best_positive_percent"
                )

                top_two_average_percent = match.get(
                    "top_two_average_percent"
                )

                if reference_count > 0:
                    st.caption(
                        "**Reference agreement: "
                        f"{references_agreed}/"
                        f"{reference_count}**"
                    )

                if best_positive_percent is not None:
                    st.caption(
                        "Best positive: "
                        f"{float(best_positive_percent):.2f}%"
                    )

                if top_two_average_percent is not None:
                    st.caption(
                        "Top-2 average: "
                        f"{float(top_two_average_percent):.2f}%"
                    )

                best_reference_name = (
                    match.get(
                        "best_reference_name"
                    )
                )

                if best_reference_name:
                    st.info(
                        "Best reference: "
                        f"{best_reference_name}"
                    )

                reference_scores = (
                    match.get(
                        "reference_scores",
                        [],
                    )
                )

                if len(reference_scores) > 1:
                    with st.expander(
                        "All reference scores",
                        expanded=False,
                    ):
                        score_rows = [
                            {
                                "Reference": (
                                    score.get(
                                        "reference_name"
                                    )
                                ),
                                "Similarity %": (
                                    score.get(
                                        "similarity_percent"
                                    )
                                ),
                            }
                            for score
                            in reference_scores
                        ]

                        st.dataframe(
                            pd.DataFrame(
                                score_rows
                            ),
                            width="stretch",
                            hide_index=True,
                        )


@st.fragment
def display_similar_image_search():
    """
    Search recursively for visually similar images
    using one or multiple reference images.
    """

    st.subheader(
        "Find Similar Images"
    )

    st.caption(
        "Ek ya multiple reference images ke basis "
        "par folders aur subfolders me visually "
        "similar images search karo."
    )

    st.info(
        "SHA-256 exact duplicates ke liye hai. "
        "Ye feature DINOv2 visual embeddings use "
        "karta hai. Multiple references dene par "
        "har candidate ka sabhi references ke against "
        "highest similarity score use hoga."
    )

    # =====================================================
    # INPUTS
    # =====================================================

    reference_input = st.text_area(
        "Reference image paths",
        key="similar_reference_images",
        placeholder=(
            "Ek image path per line paste karo:\n"
            r"C:\Users\304758\Pictures\reference_1.jpg"
            "\n"
            r"C:\Users\304758\Pictures\reference_2.jpg"
        ),
        height=150,
        help=(
            "Safest method: har reference-image path "
            "alag line me paste karo. Single path bhi "
            "supported hai."
        ),
    )

    search_root = st.text_input(
        "Search root folder",
        key="similar_search_root",
        placeholder=(
            r"C:\Users\304758\Pictures"
        ),
    )

    threshold = st.slider(
        "Minimum similarity",
        min_value=0.40,
        max_value=0.99,
        value=0.80,
        step=0.01,
        key="similar_threshold",
    )

    maximum_results = st.number_input(
        "Maximum results",
        min_value=1,
        max_value=500,
        value=100,
        step=10,
        key="similar_max_results",
    )

    st.markdown(
        "### Multi-Reference Matching"
    )

    scoring_mode_label = st.selectbox(
        "Search mode",
        options=[
            "High Recall",
            "Balanced",
            "Strict",
        ],
        index=0,
        key="similar_scoring_mode",
        help=(
            "High Recall maximum genuine matches find "
            "karne ke liye. Balanced false positives "
            "aur recall ko balance karta hai. Strict "
            "strong multi-reference agreement maangta hai."
        ),
    )

    scoring_mode_map = {
        "High Recall": "high_recall",
        "Balanced": "balanced",
        "Strict": "strict",
    }

    positive_scoring_mode = (
        scoring_mode_map[
            scoring_mode_label
        ]
    )

    if positive_scoring_mode == "high_recall":
        default_agreement_threshold = 0.45
        default_minimum_agreement = 1
    elif positive_scoring_mode == "balanced":
        default_agreement_threshold = 0.50
        default_minimum_agreement = 2
    else:
        default_agreement_threshold = 0.55
        default_minimum_agreement = 3

    positive_agreement_threshold = st.slider(
        "Reference agreement threshold",
        min_value=0.30,
        max_value=0.90,
        value=float(
            default_agreement_threshold
        ),
        step=0.01,
        key=(
            "similar_positive_agreement_threshold_"
            f"{positive_scoring_mode}"
        ),
        help=(
            "Ek reference candidate se agree tab "
            "mana jayega jab cosine similarity is "
            "threshold ke equal ya above ho."
        ),
    )

    minimum_reference_agreement = st.number_input(
        "Minimum references agreeing",
        min_value=1,
        max_value=20,
        value=int(
            default_minimum_agreement
        ),
        step=1,
        key=(
            "similar_minimum_reference_agreement_"
            f"{positive_scoring_mode}"
        ),
        help=(
            "Candidate ko minimum itni positive "
            "reference images ke saath agreement "
            "dikhana hoga. Backend available reference "
            "count ke according safely clamp karega."
        ),
    )

    if positive_scoring_mode == "high_recall":
        st.info(
            "High Recall: genuine matches miss na hon, "
            "isliye strongest reference ko zyada weight "
            "milta hai. Tumhare 25-26 out of 26 retrieval "
            "target ke liye recommended starting mode."
        )
    elif positive_scoring_mode == "balanced":
        st.info(
            "Balanced: multiple references ke agreement "
            "ko stronger importance deta hai aur false "
            "positives ko reduce karta hai."
        )
    else:
        st.info(
            "Strict: strong multi-reference agreement "
            "required. Accuracy higher ho sakti hai, "
            "lekin unusual-angle genuine images miss "
            "hone ka risk bhi higher hai."
        )

    st.markdown(
        "### Image Quality Filter"
    )

    exclude_blurry_images = st.checkbox(
        "Exclude blurry images",
        value=True,
        key="similar_exclude_blurry",
        help=(
            "Blurry images ko DINOv2 comparison "
            "se pehle skip karega."
        ),
    )

    blur_threshold = st.number_input(
        "Blur threshold",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=10.0,
        key="similar_blur_threshold",
        disabled=not exclude_blurry_images,
        help=(
            "Variance of Laplacian sharpness score. "
            "Lower score usually means more blur."
        ),
    )

    st.caption(
        "100 starting threshold hai. Actual telecom "
        "images ke results dekh kar isko calibrate "
        "kar sakte hain."
    )

    st.markdown(
        "### Optional Exclusion References"
    )

    exclude_negative_matches = st.checkbox(
        "Exclude specific / wrong image types",
        value=False,
        key="similar_exclude_negative",
        help=(
            "Optional feature. Agar candidate kisi "
            "negative reference ke zyada close hai, "
            "candidate result se reject ho jayega."
        ),
    )

    negative_reference_input = st.text_area(
        "Negative reference image paths",
        key="similar_negative_reference_images",
        placeholder=(
            "Har unwanted image ka path alag line me paste karo."
        ),
        height=110,
        disabled=not exclude_negative_matches,
        help=(
            "Ye wo example images hain jo final "
            "results me nahi chahiye."
        ),
    )

    negative_similarity_threshold = st.slider(
        "Negative similarity threshold",
        min_value=0.40,
        max_value=0.99,
        value=0.75,
        step=0.01,
        key="similar_negative_threshold",
        disabled=not exclude_negative_matches,
        help=(
            "Candidate ka negative-reference similarity "
            "is value se kam hua to negative rule "
            "candidate ko reject nahi karega."
        ),
    )

    negative_similarity_margin = st.slider(
        "Negative rejection margin",
        min_value=0.00,
        max_value=0.30,
        value=0.03,
        step=0.01,
        key="similar_negative_margin",
        disabled=not exclude_negative_matches,
        help=(
            "Negative match ko best positive match se "
            "kam se kam itna stronger hona chahiye."
        ),
    )

    st.caption(
        "Recommended starting point: negative threshold "
        "0.75 and margin 0.03. Negative references "
        "completely optional hain."
    )

    # =====================================================
    # PARSE AND PREVIEW REFERENCES
    # =====================================================

    parsed_reference_values = (
        parse_reference_image_paths(
            reference_input
        )
    )

    parsed_negative_reference_values = []

    if exclude_negative_matches:
        parsed_negative_reference_values = (
            parse_reference_image_paths(
                negative_reference_input
            )
        )

    (
        preview_negative_reference_paths,
        preview_negative_reference_errors,
    ) = validate_reference_paths(
        parsed_negative_reference_values
    )

    (
        preview_reference_paths,
        preview_reference_errors,
    ) = validate_reference_paths(
        parsed_reference_values
    )

    if parsed_reference_values:
        summary_columns = st.columns(
            3
        )

        summary_columns[0].metric(
            "Paths Entered",
            len(
                parsed_reference_values
            ),
        )

        summary_columns[1].metric(
            "Valid References",
            len(
                preview_reference_paths
            ),
        )

        summary_columns[2].metric(
            "Invalid References",
            len(
                preview_reference_errors
            ),
        )

    if preview_reference_errors:
        with st.expander(
            "Invalid reference paths",
            expanded=True,
        ):
            st.dataframe(
                pd.DataFrame(
                    preview_reference_errors
                ),
                width="stretch",
                hide_index=True,
            )

    display_reference_previews(
        preview_reference_paths
    )

    if exclude_negative_matches:

        if not parsed_negative_reference_values:
            st.warning(
                "Negative filtering ON hai, lekin koi "
                "negative reference image nahi di gayi."
            )

        if preview_negative_reference_errors:
            st.error(
                "Negative reference paths have errors."
            )

            with st.expander(
                "Invalid negative reference paths",
                expanded=True,
            ):
                st.dataframe(
                    pd.DataFrame(
                        preview_negative_reference_errors
                    ),
                    width="stretch",
                    hide_index=True,
                )

        if preview_negative_reference_paths:
            st.markdown(
                "### Exclusion Reference Images"
            )

            st.caption(
                f"{len(preview_negative_reference_paths)} "
                "negative reference image(s) selected."
            )

            columns_per_row = 5

            for start_index in range(
                0,
                len(preview_negative_reference_paths),
                columns_per_row,
            ):
                current_negative_paths = (
                    preview_negative_reference_paths[
                        start_index:
                        start_index + columns_per_row
                    ]
                )

                negative_columns = st.columns(
                    len(current_negative_paths)
                )

                for column, negative_path in zip(
                    negative_columns,
                    current_negative_paths,
                ):
                    with column:
                        st.image(
                            str(negative_path),
                            width=150,
                        )

                        st.caption(
                            negative_path.name
                        )

    # =====================================================
    # SEARCH
    # =====================================================

    if st.button(
        "Find Similar Images",
        type="primary",
        key="run_similar_search",
    ):
        clean_root = (
            str(search_root)
            .strip()
            .strip('"')
            .strip("'")
        )

        if not parsed_reference_values:
            st.error(
                "At least one reference image "
                "path required hai."
            )
            return

        if preview_reference_errors:
            st.error(
                "Search start nahi hui. Pehle invalid "
                "reference paths correct/remove karo."
            )
            return

        if not preview_reference_paths:
            st.error(
                "Koi valid reference image nahi mili."
            )
            return

        if not clean_root:
            st.error(
                "Search root folder required hai."
            )
            return

        if (
            exclude_negative_matches
            and not preview_negative_reference_paths
        ):
            st.error(
                "Negative filtering enabled but no valid "
                "negative reference image mili."
            )
            return

        if (
            exclude_negative_matches
            and preview_negative_reference_errors
        ):
            st.error(
                "Negative reference paths correct/remove "
                "karne ke baad search start karo."
            )
            return

        try:
            root_path = (
                Path(clean_root)
                .expanduser()
                .resolve()
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            st.error(
                "Invalid search root folder: "
                f"{exc}"
            )
            return

        if not root_path.is_dir():
            st.error(
                "Search root folder not found: "
                f"{root_path}"
            )
            return

        try:
            with st.spinner(
                "Loading DINOv2 and scanning images..."
            ):
                finder = (
                    get_similarity_model()
                )

                result = finder.search(
                    reference_images=(
                        preview_reference_paths
                    ),
                    search_root=root_path,
                    minimum_similarity=(
                        threshold
                    ),
                    maximum_results=int(
                        maximum_results
                    ),
                    exclude_blurry_images=(
                        exclude_blurry_images
                    ),
                    blur_threshold=float(
                        blur_threshold
                    ),
                    negative_reference_images=(
                        preview_negative_reference_paths
                        if exclude_negative_matches
                        else None
                    ),
                    exclude_negative_matches=(
                        exclude_negative_matches
                    ),
                    negative_similarity_threshold=float(
                        negative_similarity_threshold
                    ),
                    negative_similarity_margin=float(
                        negative_similarity_margin
                    ),
                    positive_scoring_mode=(
                        positive_scoring_mode
                    ),
                    positive_agreement_threshold=float(
                        positive_agreement_threshold
                    ),
                    minimum_reference_agreement=int(
                        minimum_reference_agreement
                    ),
                )

            st.session_state[
                "similar_search_result"
            ] = result

            st.success(
                "Similarity search completed using "
                f"{len(preview_reference_paths)} "
                "reference image(s)."
            )

        except SimilarImageFinderError as exc:
            st.error(
                "Similarity search failed: "
                f"{exc}"
            )
            return

        except Exception as exc:
            st.error(
                "Unexpected similarity search error: "
                f"{exc}"
            )
            return

    # =====================================================
    # RESULTS
    # =====================================================

    result = st.session_state.get(
        "similar_search_result"
    )

    if not isinstance(
        result,
        dict,
    ):
        return

    st.divider()

    summary_columns = st.columns(
        5
    )

    summary_columns[0].metric(
        "References",
        result.get(
            "reference_count",
            len(
                result.get(
                    "reference_images",
                    [],
                )
            ),
        ),
    )

    summary_columns[1].metric(
        "Images Scanned",
        result.get(
            "images_scanned",
            0,
        ),
    )

    summary_columns[2].metric(
        "Matches",
        result.get(
            "matches_found",
            0,
        ),
    )

    summary_columns[3].metric(
        "Failed",
        result.get(
            "failed_count",
            0,
        ),
    )

    summary_columns[4].metric(
        "Device",
        result.get(
            "device",
            "unknown",
        ),
    )

    st.caption(
        "Scoring method: "
        f"{result.get('scoring_method', 'cosine similarity')}"
    )

    scoring_columns = st.columns(
        3
    )

    scoring_columns[0].metric(
        "Search Mode",
        str(
            result.get(
                "positive_scoring_mode",
                "high_recall",
            )
        )
        .replace("_", " ")
        .title(),
    )

    scoring_columns[1].metric(
        "Agreement Threshold",
        result.get(
            "positive_agreement_threshold",
            0,
        ),
    )

    scoring_columns[2].metric(
        "Minimum Agreement",
        result.get(
            "minimum_reference_agreement",
            1,
        ),
    )

    quality_columns = st.columns(
        3
    )

    quality_columns[0].metric(
        "Blurry Skipped",
        result.get(
            "blurry_skipped",
            0,
        ),
    )

    quality_columns[1].metric(
        "DINOv2 Compared",
        result.get(
            "dinov2_compared",
            0,
        ),
    )

    quality_columns[2].metric(
        "Blur Threshold",
        result.get(
            "blur_threshold",
            0,
        ),
    )

    negative_columns = st.columns(
        3
    )

    negative_columns[0].metric(
        "Negative References",
        result.get(
            "negative_reference_count",
            0,
        ),
    )

    negative_columns[1].metric(
        "Negative Excluded",
        result.get(
            "negative_excluded_count",
            0,
        ),
    )

    negative_columns[2].metric(
        "Negative Threshold",
        result.get(
            "negative_similarity_threshold",
            0,
        ),
    )

    matches = result.get(
        "matches",
        [],
    )

    negative_excluded = result.get(
        "negative_excluded",
        [],
    )

    if negative_excluded:
        st.markdown(
            "### Images Excluded by Negative References"
        )

        st.caption(
            "Ye candidates positive references se "
            "similar the, lekin configured negative "
            "reference se stronger match hone ki wajah "
            "se final results se remove hue."
        )

        with st.expander(
            f"View {len(negative_excluded)} excluded image(s)",
            expanded=False,
        ):
            columns_per_row = 4

            for start_index in range(
                0,
                len(negative_excluded),
                columns_per_row,
            ):
                current_excluded = negative_excluded[
                    start_index:
                    start_index + columns_per_row
                ]

                excluded_columns = st.columns(
                    len(current_excluded)
                )

                for column, excluded_item in zip(
                    excluded_columns,
                    current_excluded,
                ):
                    with column:

                        excluded_path = Path(
                            excluded_item.get(
                                "image_path",
                                "",
                            )
                        )

                        if excluded_path.is_file():
                            st.image(
                                str(excluded_path),
                                width=160,
                            )

                        st.caption(
                            excluded_item.get(
                                "relative_path",
                                excluded_path.name,
                            )
                        )

                        positive_percent = float(
                            excluded_item.get(
                                "best_positive_percent",
                                0,
                            )
                        )

                        negative_percent = float(
                            excluded_item.get(
                                "best_negative_percent",
                                0,
                            )
                        )

                        st.markdown(
                            f"**Positive: "
                            f"{positive_percent:.2f}%**"
                        )

                        st.markdown(
                            f"**Negative: "
                            f"{negative_percent:.2f}%**"
                        )

                        negative_name = (
                            excluded_item.get(
                                "best_negative_reference"
                            )
                        )

                        if negative_name:
                            st.caption(
                                "Excluded by: "
                                f"{negative_name}"
                            )

                        st.warning(
                            excluded_item.get(
                                "reason",
                                "Matched exclusion reference",
                            )
                        )

    blurry_images = result.get(
        "blurry_images",
        [],
    )

    if blurry_images:
        st.markdown(
            "### Blurry Images Skipped"
        )

        st.caption(
            "Ye images quality gate ki wajah se "
            "DINOv2 comparison se pehle skip hui."
        )

        with st.expander(
            f"View {len(blurry_images)} blurry image(s)",
            expanded=False,
        ):
            columns_per_row = 5

            for start_index in range(
                0,
                len(blurry_images),
                columns_per_row,
            ):
                current_blurry = blurry_images[
                    start_index:
                    start_index + columns_per_row
                ]

                columns = st.columns(
                    len(current_blurry)
                )

                for column, blurry_item in zip(
                    columns,
                    current_blurry,
                ):
                    with column:
                        blurry_path = Path(
                            blurry_item.get(
                                "image_path",
                                "",
                            )
                        )

                        if blurry_path.is_file():
                            st.image(
                                str(blurry_path),
                                width=150,
                            )

                        st.caption(
                            blurry_item.get(
                                "relative_path",
                                blurry_path.name,
                            )
                        )

                        blur_score = float(
                            blurry_item.get(
                                "blur_score",
                                0,
                            )
                        )

                        st.markdown(
                            f"**Blur score: {blur_score:.2f}**"
                        )

    if not matches:
        st.warning(
            "Current filters par koi similar image "
            "nahi mili. Similarity threshold ya blur "
            "threshold adjust karke try karo."
        )
        return

    # =====================================================
    # RESULT TABLE
    # =====================================================

    table_rows = []

    for rank, match in enumerate(
        matches,
        start=1,
    ):
        table_rows.append(
            {
                "Rank": rank,
                "Final Score %": (
                    match.get(
                        "final_similarity_percent",
                        match.get(
                            "similarity_percent"
                        ),
                    )
                ),
                "Best Positive %": (
                    match.get(
                        "best_positive_percent"
                    )
                ),
                "Top-2 Average %": (
                    match.get(
                        "top_two_average_percent"
                    )
                ),
                "All Reference Average %": (
                    match.get(
                        "all_reference_average_percent"
                    )
                ),
                "Agreement": (
                    f"{match.get('references_agreed', 0)}/"
                    f"{match.get('reference_count', 0)}"
                ),
                "Agreement %": (
                    match.get(
                        "agreement_percent"
                    )
                ),
                "Blur Score": (
                    match.get(
                        "blur_score"
                    )
                ),
                "Best Reference": (
                    match.get(
                        "best_reference_name"
                    )
                ),
                "Relative Path": (
                    match.get(
                        "relative_path"
                    )
                ),
                "Full Path": (
                    match.get(
                        "image_path"
                    )
                ),
            }
        )

    st.markdown(
        "### Ranked Results"
    )

    st.dataframe(
        pd.DataFrame(
            table_rows
        ),
        width="stretch",
        hide_index=True,
    )

    # =====================================================
    # MATCH PREVIEWS
    # =====================================================

    display_match_previews(
        matches
    )

    # =====================================================
    # FAILED IMAGES
    # =====================================================

    failures = result.get(
        "failures",
        [],
    )

    if failures:
        with st.expander(
            "Images that could not be processed",
            expanded=False,
        ):
            st.dataframe(
                pd.DataFrame(
                    failures
                ),
                width="stretch",
                hide_index=True,
            )

    # =====================================================
    # COPY MATCHES
    # =====================================================

    st.divider()

    st.markdown(
        "### Copy Similar Images"
    )

    st.caption(
        "Matched images ko separate folder me copy "
        "karega. Original images modify ya delete "
        "nahi hongi."
    )

    default_output_folder = (
        Path(
            result.get(
                "search_root",
                ".",
            )
        )
        / "similar_image_results"
    )

    output_folder = st.text_input(
        "Output folder",
        value=str(
            default_output_folder
        ),
        key="similar_output_folder",
    )

    if st.button(
        "Copy Similar Images",
        key="copy_similar_matches",
    ):
        clean_output_folder = (
            str(output_folder)
            .strip()
            .strip('"')
            .strip("'")
        )

        if not clean_output_folder:
            st.error(
                "Output folder required hai."
            )
            return

        try:
            copy_report = (
                copy_similarity_matches(
                    search_result=result,
                    output_folder=(
                        clean_output_folder
                    ),
                )
            )

            st.session_state[
                "similar_copy_report"
            ] = copy_report

            if copy_report.get(
                "failed_count",
                0,
            ):
                st.warning(
                    "Copy completed with some errors. "
                    f"Copied: "
                    f"{copy_report.get('copied_count', 0)}, "
                    f"Failed: "
                    f"{copy_report.get('failed_count', 0)}"
                )
            else:
                st.success(
                    "All similar images copied: "
                    f"{copy_report.get('copied_count', 0)}"
                )

            st.code(
                copy_report.get(
                    "output_folder",
                    clean_output_folder,
                )
            )

        except Exception as exc:
            st.error(
                "Could not copy similar images: "
                f"{exc}"
            )
