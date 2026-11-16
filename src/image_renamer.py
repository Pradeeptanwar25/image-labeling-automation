from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any


class ImageRenameError(Exception):
    """Raised when image renaming cannot complete safely."""

    pass


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".jfif",
    ".bmp",
    ".webp",
}

def _number_key(
    image_path: Path,
) -> tuple[int, str]:
    numbers = re.findall(
        r"\d+",
        image_path.stem,
    )

    number = (
        int(
            numbers[-1]
        )
        if numbers
        else 999999999
    )

    return (
        number,
        image_path.name.casefold(),
    )


def _standard_pattern(
    folder_path: Path,
) -> re.Pattern:
    prefix = re.escape(
        folder_path.name
    )

    return re.compile(
        rf"^{prefix}_(\d{{6}})$",
        flags=re.IGNORECASE,
    )


def build_rename_plan(
    folder: str | Path,
) -> list[dict[str, Any]]:
    """
    Folder-prefix automatic rename plan.

    Fresh folder:
        No standardized images exist.
        All images are renamed from 000001.

    Existing standardized folder:
        Existing FolderName_###### files are protected.
        Only new/random-name images are renamed from
        MAX(existing number) + 1.
    """

    folder_path = Path(
        folder
    ).expanduser().resolve()

    if not folder_path.is_dir():
        raise ImageRenameError(
            "Folder not found: "
            f"{folder_path}"
        )

    images = sorted(
        [
            path
            for path
            in folder_path.iterdir()
            if (
                path.is_file()
                and path.suffix.casefold()
                in SUPPORTED_EXTENSIONS
            )
        ],
        key=_number_key,
    )

    if not images:
        raise ImageRenameError(
            "No supported images found: "
            f"{folder_path}"
        )

    pattern = _standard_pattern(
        folder_path
    )

    existing = []
    new_images = []

    for image_path in images:
        match = pattern.fullmatch(
            image_path.stem
        )

        if match:
            existing.append(
                (
                    int(
                        match.group(1)
                    ),
                    image_path,
                )
            )

        else:
            new_images.append(
                image_path
            )

    plan = []

    if not existing:
        # Fresh folder mode.
        images_to_rename = images
        next_number = 1
        mode = "fresh_folder"

    else:
        # Existing standardized folder mode.
        for sequence_number, image_path in sorted(
            existing,
            key=lambda item: item[0],
        ):
            plan.append(
                {
                    "source_path": image_path,
                    "temporary_path": None,
                    "destination_path": image_path,
                    "old_name": image_path.name,
                    "new_name": image_path.name,
                    "unchanged": True,
                    "protected": True,
                    "sequence_number": sequence_number,
                }
            )

        images_to_rename = sorted(
            new_images,
            key=_number_key,
        )

        next_number = max(
            number
            for number, _
            in existing
        ) + 1

        mode = "append_new_images"

    for image_path in images_to_rename:
        extension = image_path.suffix.casefold()

        new_name = (
            f"{folder_path.name}_"
            f"{next_number:06d}"
            f"{extension}"
        )

        destination_path = (
            folder_path
            / new_name
        )

        unchanged = (
            image_path.name
            == new_name
        )

        temporary_path = None

        if not unchanged:
            temporary_path = (
                folder_path
                / (
                    ".__rename_"
                    + uuid.uuid4().hex
                    + extension
                )
            )

        plan.append(
            {
                "source_path": image_path,
                "temporary_path": temporary_path,
                "destination_path": destination_path,
                "old_name": image_path.name,
                "new_name": new_name,
                "unchanged": unchanged,
                "protected": False,
                "sequence_number": next_number,
                "mode": mode,
            }
        )

        next_number += 1

    return plan


def preview_rename_plan(
    folder: str | Path,
) -> dict[str, Any]:
    folder_path = Path(
        folder
    ).expanduser().resolve()

    plan = build_rename_plan(
        folder_path
    )

    changed = [
        item
        for item
        in plan
        if not item[
            "unchanged"
        ]
    ]

    protected = [
        item
        for item
        in plan
        if item.get(
            "protected",
            False,
        )
    ]

    mode = (
        "append_new_images"
        if protected
        else "fresh_folder"
    )

    return {
        "folder": str(
            folder_path
        ),
        "folder_prefix": folder_path.name,
        "mode": mode,
        "total_images": len(
            plan
        ),
        "protected_images": len(
            protected
        ),
        "files_to_rename": len(
            changed
        ),
        "renamed": [
            {
                "old_name": item[
                    "old_name"
                ],
                "new_name": item[
                    "new_name"
                ],
                "unchanged": item[
                    "unchanged"
                ],
                "protected": item.get(
                    "protected",
                    False,
                ),
            }
            for item
            in plan
        ],
    }


def rename_images_with_folder_prefix(
    folder: str | Path,
) -> dict[str, Any]:
    folder_path = Path(
        folder
    ).expanduser().resolve()

    plan = build_rename_plan(
        folder_path
    )

    changed = [
        item
        for item
        in plan
        if not item[
            "unchanged"
        ]
    ]

    protected_count = sum(
        1
        for item
        in plan
        if item.get(
            "protected",
            False,
        )
    )

    mode = (
        "append_new_images"
        if protected_count
        else "fresh_folder"
    )

    if not changed:
        return {
            "status": "completed",
            "folder": str(
                folder_path
            ),
            "mode": mode,
            "protected_images": protected_count,
            "renamed_count": 0,
            "renamed": [],
        }

    moved_to_temp = []
    moved_to_final = []

    try:
        for item in changed:
            temporary = item[
                "temporary_path"
            ]

            if temporary is None:
                raise ImageRenameError(
                    "Temporary path missing for "
                    f"{item['old_name']}"
                )

            os.replace(
                item[
                    "source_path"
                ],
                temporary,
            )

            moved_to_temp.append(
                item
            )

        for item in changed:
            destination = item[
                "destination_path"
            ]

            if destination.exists():
                raise ImageRenameError(
                    "Destination already exists: "
                    f"{destination}"
                )

            os.replace(
                item[
                    "temporary_path"
                ],
                destination,
            )

            moved_to_final.append(
                item
            )

    except Exception as exc:
        for item in reversed(
            moved_to_final
        ):
            try:
                destination = item[
                    "destination_path"
                ]

                original = item[
                    "source_path"
                ]

                if (
                    destination.exists()
                    and not original.exists()
                ):
                    os.replace(
                        destination,
                        original,
                    )

            except OSError:
                pass

        for item in reversed(
            moved_to_temp
        ):
            try:
                temporary = item[
                    "temporary_path"
                ]

                original = item[
                    "source_path"
                ]

                if (
                    temporary is not None
                    and temporary.exists()
                    and not original.exists()
                ):
                    os.replace(
                        temporary,
                        original,
                    )

            except OSError:
                pass

        raise ImageRenameError(
            "Image rename failed: "
            f"{exc}"
        ) from exc

    return {
        "status": "completed",
        "folder": str(
            folder_path
        ),
        "mode": mode,
        "protected_images": protected_count,
        "renamed_count": len(
            changed
        ),
        "renamed": [
            {
                "old_name": item[
                    "old_name"
                ],
                "new_name": item[
                    "new_name"
                ],
            }
            for item
            in changed
        ],
    }
