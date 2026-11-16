from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from src.safe_io import (
    SafeIOError,
    safe_remove_file,
)


class DuplicateManagerError(Exception):
    """Raised when duplicate deletion cannot complete safely."""

    pass


def _resolve_reported_path(
    dataset_root: Path,
    relative_path: str,
) -> Path:
    reported_path = Path(relative_path)
    if reported_path.is_absolute():
        raise ValueError(
            "Reported path must be relative to the dataset root."
        )

    target = (
        dataset_root
        / reported_path
    ).resolve()

    try:
        target.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            "Safety check failed: file is outside the dataset root."
        ) from exc

    return target


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_pattern(
    dataset_root: Path,
) -> re.Pattern:
    """
    Existing standardized files are protected.

    Example for S18:

        S18_000001.jpg
        S18_000173.jpeg
    """

    prefix = re.escape(
        dataset_root.name
    )

    return re.compile(
        rf"^{prefix}_(\d{{6}})$",
        flags=re.IGNORECASE,
    )


def is_protected_image(
    image_path: str | Path,
    dataset_root: str | Path,
) -> bool:
    """
    Return True when an image already has the
    standardized folder-based filename.
    """

    path = Path(
        image_path
    )

    root = Path(
        dataset_root
    )

    pattern = _protected_pattern(
        root
    )

    return bool(
        pattern.fullmatch(
            path.stem
        )
    )


def _retention_priority(
    record: dict[str, Any],
    dataset_root: Path,
) -> tuple[int, str]:
    """
    Protected standardized files receive priority 0.

    New/random-name files receive priority 1.

    Therefore, when an old labelled image and a new
    duplicate have the same SHA-256, the old labelled
    image is retained.
    """

    relative_path = Path(
        str(
            record.get(
                "relative_path",
                "",
            )
        )
    )

    protected = is_protected_image(
        relative_path,
        dataset_root,
    )

    return (
        0 if protected else 1,
        relative_path.as_posix().casefold(),
    )


def get_root_duplicate_groups(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Return exact SHA-256 duplicate groups from the
    currently scanned dataset root.

    Retention rules:

    1. Existing FolderName_###### images are protected.
    2. A protected image is always retained over a new
       random-name duplicate.
    3. If all files are new, the first sorted file is kept.
    4. Existing labelled filenames are never renamed or
       removed merely because a new duplicate was added.
    """

    dataset_root = Path(
        str(
            report[
                "root"
            ]
        )
    ).expanduser().resolve()

    groups_by_hash: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for record in report.get(
        "records",
        [],
    ):
        if record.get(
            "status"
        ) != "ok":
            continue

        image_hash = str(
            record.get(
                "sha256",
                "",
            )
        ).strip()

        relative_path = Path(
            str(
                record.get(
                    "relative_path",
                    "",
                )
            )
        ).as_posix()

        if not image_hash or not relative_path:
            continue

        groups_by_hash.setdefault(
            image_hash,
            [],
        ).append(
            {
                **record,
                "relative_path": relative_path,
            }
        )

    duplicate_groups = []

    for image_hash, records in groups_by_hash.items():
        if len(
            records
        ) < 2:
            continue

        ordered = sorted(
            records,
            key=lambda record: (
                _retention_priority(
                    record,
                    dataset_root,
                )
            ),
        )

        retained = ordered[0]
        duplicates = ordered[1:]

        duplicate_groups.append(
            {
                "sha256": image_hash,
                "retained": retained,
                "retained_is_protected": (
                    is_protected_image(
                        retained[
                            "relative_path"
                        ],
                        dataset_root,
                    )
                ),
                "duplicates": duplicates,
            }
        )

    return sorted(
        duplicate_groups,
        key=lambda group: (
            str(
                group[
                    "retained"
                ][
                    "relative_path"
                ]
            ).casefold()
        ),
    )


def get_root_duplicate_summary(
    report: dict[str, Any],
) -> dict[str, int]:
    groups = get_root_duplicate_groups(
        report
    )

    files_to_delete = sum(
        len(
            group[
                "duplicates"
            ]
        )
        for group
        in groups
    )

    protected_retained = sum(
        1
        for group
        in groups
        if group[
            "retained_is_protected"
        ]
    )

    return {
        "duplicate_groups": len(
            groups
        ),
        "files_to_delete": (
            files_to_delete
        ),
        "files_to_retain": len(
            groups
        ),
        "protected_files_retained": (
            protected_retained
        ),
    }


def delete_root_duplicates(
    report: dict[str, Any],
) -> dict[str, Any]:
    """
    Delete extra exact SHA-256 copies.

    Protected existing files are always retained
    when duplicates are newly added.
    """

    dataset_root = Path(
        str(
            report[
                "root"
            ]
        )
    ).expanduser().resolve()

    if not dataset_root.is_dir():
        raise DuplicateManagerError(
            "Dataset root not found: "
            f"{dataset_root}"
        )

    groups = get_root_duplicate_groups(
        report
    )

    deleted_files = []
    retained_files = []
    protected_retained_files = []
    failed_files = []

    for group in groups:
        retained_record = group[
            "retained"
        ]

        retained_name = retained_record[
            "relative_path"
        ]

        retained_files.append(
            retained_name
        )

        if group[
            "retained_is_protected"
        ]:
            protected_retained_files.append(
                retained_name
            )

        expected_hash = str(
            group.get("sha256", "")
        ).casefold()
        group_paths: list[tuple[str, Path]] = []
        group_is_current = bool(expected_hash)

        records = [
            retained_record,
            *group["duplicates"],
        ]
        for record in records:
            relative_name = str(
                record.get("relative_path", "")
            )

            try:
                target = _resolve_reported_path(
                    dataset_root,
                    relative_name,
                )
            except ValueError as exc:
                failed_files.append(
                    {
                        "file": relative_name,
                        "error": str(exc),
                    }
                )
                group_is_current = False
                continue

            if not target.is_file():
                failed_files.append(
                    {
                        "file": relative_name,
                        "error": "File no longer exists.",
                    }
                )
                group_is_current = False
                continue

            try:
                current_hash = _sha256_file(target)
            except OSError as exc:
                failed_files.append(
                    {
                        "file": relative_name,
                        "error": f"Could not verify file hash: {exc}",
                    }
                )
                group_is_current = False
                continue

            if current_hash != expected_hash:
                failed_files.append(
                    {
                        "file": relative_name,
                        "error": (
                            "File changed since the duplicate scan; "
                            "this duplicate group was not deleted."
                        ),
                    }
                )
                group_is_current = False
                continue

            group_paths.append(
                (relative_name, target)
            )

        if not group_is_current:
            continue

        retained_target = group_paths[0][1]
        for relative_name, target in group_paths[1:]:
            try:
                if (
                    not retained_target.is_file()
                    or _sha256_file(retained_target)
                    != expected_hash
                ):
                    failed_files.append(
                        {
                            "file": retained_name,
                            "error": (
                                "Retained file changed since the "
                                "duplicate scan; remaining duplicates "
                                "were not deleted."
                            ),
                        }
                    )
                    break

                if (
                    not target.is_file()
                    or _sha256_file(target) != expected_hash
                ):
                    failed_files.append(
                        {
                            "file": relative_name,
                            "error": (
                                "File changed since the duplicate scan; "
                                "it and remaining duplicates were not "
                                "deleted."
                            ),
                        }
                    )
                    break

                safe_remove_file(
                    target,
                    retries=20,
                )
                deleted_files.append(
                    Path(relative_name).as_posix()
                )
            except OSError as exc:
                failed_files.append(
                    {
                        "file": relative_name,
                        "error": f"Could not verify file hash: {exc}",
                    }
                )
                break
            except SafeIOError as exc:
                failed_files.append(
                    {
                        "file": relative_name,
                        "error": str(exc),
                    }
                )
                break

    return {
        "dataset_root": str(
            dataset_root
        ),
        "duplicate_groups": len(
            groups
        ),
        "deleted_count": len(
            deleted_files
        ),
        "retained_count": len(
            retained_files
        ),
        "protected_retained_count": len(
            protected_retained_files
        ),
        "failed_count": len(
            failed_files
        ),
        "deleted_files": deleted_files,
        "retained_files": retained_files,
        "protected_retained_files": (
            protected_retained_files
        ),
        "failed_files": failed_files,
    }
