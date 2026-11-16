from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any


class ImageCollectorError(Exception):
    """Raised when image collection cannot complete safely."""

    pass


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".jfif",
}


CATEGORY_PATTERN = re.compile(
    r"^s\d+(?:[_-].+)?$",
    flags=re.IGNORECASE,
)


def _is_category_folder(
    folder_name: str,
) -> bool:
    """
    Examples accepted:

        S5
        S20
        S20_A1
        S26_3
        s22-4
    """

    return bool(
        CATEGORY_PATTERN.fullmatch(
            str(
                folder_name
            ).strip()
        )
    )


def _find_category_path(
    image_path: Path,
    source_root: Path,
) -> Path | None:
    """
    Find the nearest category folder in the image's
    parent hierarchy.

    Example:

        source/site/classified/S20/S20_A1/image.jpg

    returns:

        S20/S20_A1
    """

    try:
        relative_parent = (
            image_path.parent
            .relative_to(
                source_root
            )
        )

    except ValueError:
        return None

    parts = list(
        relative_parent.parts
    )

    category_index = None

    for index, part in enumerate(
        parts
    ):
        if _is_category_folder(
            part
        ):
            category_index = index
            break

    if category_index is None:
        return None

    return Path(
        *parts[
            category_index:
        ]
    )


def _collision_safe_destination(
    destination: Path,
) -> Path:
    """
    Avoid overwriting an existing output image.

    Example:

        image.jpg
        image_1.jpg
        image_2.jpg
    """

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix

    counter = 1

    while True:
        candidate = destination.with_name(
            f"{stem}_{counter}{suffix}"
        )

        if not candidate.exists():
            return candidate

        counter += 1


def build_collection_plan(
    source_folder: str | Path,
    output_folder: str | Path,
) -> dict[str, Any]:
    """
    Build a preview plan without copying files.
    """

    source_root = Path(
        source_folder
    ).expanduser().resolve()

    output_root = Path(
        output_folder
    ).expanduser().resolve()

    if not source_root.is_dir():
        raise ImageCollectorError(
            "Source folder not found: "
            f"{source_root}"
        )

    if output_root == source_root:
        raise ImageCollectorError(
            "Output folder cannot be the same "
            "as the source folder."
        )

    try:
        output_root.relative_to(
            source_root
        )

    except ValueError:
        output_is_inside_source = False

    else:
        output_is_inside_source = True

    records = []
    skipped_without_category = []

    for image_path in sorted(
        source_root.rglob("*"),
        key=lambda path: (
            path.as_posix().casefold()
        ),
    ):
        if not image_path.is_file():
            continue

        if (
            image_path.suffix.casefold()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        if (
            output_is_inside_source
            and output_root in image_path.parents
        ):
            continue

        category_path = _find_category_path(
            image_path,
            source_root,
        )

        if category_path is None:
            skipped_without_category.append(
                image_path
                .relative_to(
                    source_root
                )
                .as_posix()
            )

            continue

        proposed_destination = (
            output_root
            / category_path
            / image_path.name
        )

        records.append(
            {
                "source_path": str(
                    image_path
                ),
                "source_relative": (
                    image_path
                    .relative_to(
                        source_root
                    )
                    .as_posix()
                ),
                "category_path": (
                    category_path
                    .as_posix()
                ),
                "proposed_destination": str(
                    proposed_destination
                ),
                "image_name": (
                    image_path.name
                ),
            }
        )

    if not records:
        raise ImageCollectorError(
            "No categorized images were found. "
            "Expected category folders such as "
            "S20, S20_A1 or S26_3."
        )

    categories = sorted(
        {
            record[
                "category_path"
            ]
            for record
            in records
        },
        key=str.casefold,
    )

    return {
        "source_folder": str(
            source_root
        ),
        "output_folder": str(
            output_root
        ),
        "image_count": len(
            records
        ),
        "category_count": len(
            categories
        ),
        "categories": (
            categories
        ),
        "skipped_without_category_count": len(
            skipped_without_category
        ),
        "skipped_without_category": (
            skipped_without_category
        ),
        "records": (
            records
        ),
    }


def collect_classified_images(
    source_folder: str | Path,
    output_folder: str | Path,
) -> dict[str, Any]:
    """
    Copy categorized images into one consolidated
    output structure.

    Source files are never modified or deleted.
    """

    plan = build_collection_plan(
        source_folder=source_folder,
        output_folder=output_folder,
    )

    output_root = Path(
        plan[
            "output_folder"
        ]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied_files = []
    failed_files = []

    for record in plan[
        "records"
    ]:
        source_path = Path(
            record[
                "source_path"
            ]
        )

        category_path = Path(
            record[
                "category_path"
            ]
        )

        destination_directory = (
            output_root
            / category_path
        )

        destination_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        requested_destination = (
            destination_directory
            / source_path.name
        )

        destination_path = (
            _collision_safe_destination(
                requested_destination
            )
        )

        try:
            shutil.copy2(
                source_path,
                destination_path,
            )

            copied_files.append(
                {
                    "source": str(
                        source_path
                    ),
                    "destination": str(
                        destination_path
                    ),
                    "category_path": (
                        category_path
                        .as_posix()
                    ),
                    "renamed_for_collision": (
                        destination_path.name
                        != source_path.name
                    ),
                }
            )

        except OSError as exc:
            failed_files.append(
                {
                    "source": str(
                        source_path
                    ),
                    "error": str(
                        exc
                    ),
                }
            )

    return {
        "status": (
            "completed"
            if not failed_files
            else "completed_with_errors"
        ),
        "source_folder": (
            plan[
                "source_folder"
            ]
        ),
        "output_folder": (
            plan[
                "output_folder"
            ]
        ),
        "categories_found": (
            plan[
                "category_count"
            ]
        ),
        "images_discovered": (
            plan[
                "image_count"
            ]
        ),
        "images_copied": len(
            copied_files
        ),
        "images_failed": len(
            failed_files
        ),
        "skipped_without_category": (
            plan[
                "skipped_without_category_count"
            ]
        ),
        "copied_files": (
            copied_files
        ),
        "failed_files": (
            failed_files
        ),
    }
