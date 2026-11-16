from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SimilarityFeedbackError(Exception):
    """Raised when similarity feedback cannot be stored."""

    pass


VALID_LABELS = {
    "correct",
    "wrong",
    "skip",
}


class SimilarityFeedbackStore:
    """
    Persistent feedback store for similarity-search results.

    Labels:
        correct -> desired image
        wrong   -> visually similar but unwanted image
        skip    -> do not use for classifier training

    Original image files are never modified.
    """

    def __init__(
        self,
        store_path: str | Path = (
            "data/similarity_feedback/feedback.json"
        ),
    ) -> None:
        self.store_path = (
            Path(store_path)
            .expanduser()
            .resolve()
        )

        self.store_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _load(self) -> dict[str, Any]:
        if not self.store_path.is_file():
            return {
                "version": 1,
                "records": {},
            }

        try:
            raw = self.store_path.read_text(
                encoding="utf-8-sig"
            )

            data = json.loads(raw)

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise SimilarityFeedbackError(
                "Could not load similarity feedback: "
                f"{exc}"
            ) from exc

        if not isinstance(data, dict):
            raise SimilarityFeedbackError(
                "Similarity feedback root must be "
                "a JSON object."
            )

        records = data.get(
            "records",
            {},
        )

        if not isinstance(records, dict):
            raise SimilarityFeedbackError(
                "Similarity feedback records must "
                "be a JSON object."
            )

        data.setdefault(
            "version",
            1,
        )

        data["records"] = records

        return data

    def _save_atomic(
        self,
        data: dict[str, Any],
    ) -> None:
        """
        Save safely using temporary-file replacement.

        This avoids partially-written JSON when a write
        is interrupted.
        """

        self.store_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path: Path | None = None

        try:
            fd, temp_name = tempfile.mkstemp(
                prefix=(
                    self.store_path.stem
                    + "_"
                ),
                suffix=".tmp",
                dir=str(
                    self.store_path.parent
                ),
            )

            temp_path = Path(
                temp_name
            )

            with os.fdopen(
                fd,
                "w",
                encoding="utf-8-sig",
                newline="\n",
            ) as handle:
                json.dump(
                    data,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

                handle.flush()

                os.fsync(
                    handle.fileno()
                )

            os.replace(
                temp_path,
                self.store_path,
            )

        except OSError as exc:
            if (
                temp_path is not None
                and temp_path.exists()
            ):
                try:
                    temp_path.unlink()
                except OSError:
                    pass

            raise SimilarityFeedbackError(
                "Could not save similarity feedback: "
                f"{exc}"
            ) from exc

    @staticmethod
    def _normalize_image_path(
        image_path: str | Path,
    ) -> Path:
        path = (
            Path(image_path)
            .expanduser()
            .resolve()
        )

        if not path.is_file():
            raise SimilarityFeedbackError(
                f"Image not found: {path}"
            )

        return path

    def set_label(
        self,
        image_path: str | Path,
        label: str,
        *,
        similarity_result: (
            dict[str, Any] | None
        ) = None,
    ) -> dict[str, Any]:
        """
        Save or update one image label.
        """

        normalized_label = (
            str(label)
            .strip()
            .casefold()
        )

        if normalized_label not in VALID_LABELS:
            raise SimilarityFeedbackError(
                "Feedback label must be one of: "
                "correct, wrong, skip."
            )

        path = self._normalize_image_path(
            image_path
        )

        data = self._load()

        key = str(
            path
        ).casefold()

        existing = (
            data["records"]
            .get(
                key,
                {},
            )
        )

        record = {
            "image_path": str(
                path
            ),
            "image_name": path.name,
            "label": normalized_label,
            "updated_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

        if existing.get(
            "created_at_utc"
        ):
            record[
                "created_at_utc"
            ] = existing[
                "created_at_utc"
            ]
        else:
            record[
                "created_at_utc"
            ] = record[
                "updated_at_utc"
            ]

        if similarity_result:
            record[
                "final_similarity"
            ] = similarity_result.get(
                "final_similarity",
                similarity_result.get(
                    "similarity"
                ),
            )

            record[
                "best_positive_similarity"
            ] = similarity_result.get(
                "best_positive_similarity"
            )

            record[
                "top_two_average"
            ] = similarity_result.get(
                "top_two_average"
            )

            record[
                "all_reference_average"
            ] = similarity_result.get(
                "all_reference_average"
            )

            record[
                "references_agreed"
            ] = similarity_result.get(
                "references_agreed"
            )

            record[
                "reference_count"
            ] = similarity_result.get(
                "reference_count"
            )

            record[
                "blur_score"
            ] = similarity_result.get(
                "blur_score"
            )

        data[
            "records"
        ][key] = record

        self._save_atomic(
            data
        )

        return record

    def set_labels_batch(
        self,
        updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Save multiple feedback labels using one JSON
        read and one atomic JSON write.

        Each update must contain:
            image_path
            label

        Optional:
            similarity_result
        """

        if not isinstance(
            updates,
            list,
        ):
            raise SimilarityFeedbackError(
                "Batch updates must be a list."
            )

        data = self._load()

        saved_records: list[
            dict[str, Any]
        ] = []

        for update in updates:
            if not isinstance(
                update,
                dict,
            ):
                raise SimilarityFeedbackError(
                    "Every batch update must "
                    "be a dictionary."
                )

            image_path = update.get(
                "image_path"
            )

            normalized_label = (
                str(
                    update.get(
                        "label",
                        "",
                    )
                )
                .strip()
                .casefold()
            )

            if normalized_label not in VALID_LABELS:
                raise SimilarityFeedbackError(
                    "Feedback label must be one of: "
                    "correct, wrong, skip."
                )

            path = self._normalize_image_path(
                image_path
            )

            key = str(
                path
            ).casefold()

            existing = (
                data["records"]
                .get(
                    key,
                    {},
                )
            )

            now_utc = datetime.now(
                timezone.utc
            ).isoformat()

            record = {
                "image_path": str(
                    path
                ),
                "image_name": path.name,
                "label": normalized_label,
                "created_at_utc": (
                    existing.get(
                        "created_at_utc",
                        now_utc,
                    )
                ),
                "updated_at_utc": now_utc,
            }

            similarity_result = update.get(
                "similarity_result"
            )

            if isinstance(
                similarity_result,
                dict,
            ):
                fields = [
                    "final_similarity",
                    "best_positive_similarity",
                    "top_two_average",
                    "all_reference_average",
                    "references_agreed",
                    "reference_count",
                    "blur_score",
                ]

                for field_name in fields:
                    value = similarity_result.get(
                        field_name
                    )

                    if (
                        field_name
                        == "final_similarity"
                        and value is None
                    ):
                        value = similarity_result.get(
                            "similarity"
                        )

                    record[
                        field_name
                    ] = value

            data[
                "records"
            ][key] = record

            saved_records.append(
                record
            )

        if saved_records:
            self._save_atomic(
                data
            )

        summary = self.summary()

        return {
            "saved_count": len(
                saved_records
            ),
            "records": saved_records,
            "summary": summary,
        }

    def remove_label(
        self,
        image_path: str | Path,
    ) -> bool:
        path = (
            Path(image_path)
            .expanduser()
            .resolve()
        )

        key = str(
            path
        ).casefold()

        data = self._load()

        if key not in data[
            "records"
        ]:
            return False

        del data[
            "records"
        ][key]

        self._save_atomic(
            data
        )

        return True

    def get_label(
        self,
        image_path: str | Path,
    ) -> str | None:
        path = (
            Path(image_path)
            .expanduser()
            .resolve()
        )

        key = str(
            path
        ).casefold()

        record = (
            self._load()[
                "records"
            ]
            .get(
                key
            )
        )

        if not record:
            return None

        return record.get(
            "label"
        )

    def records(
        self,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        values = list(
            self._load()[
                "records"
            ].values()
        )

        if label is not None:
            normalized = (
                str(label)
                .strip()
                .casefold()
            )

            values = [
                record
                for record in values
                if record.get(
                    "label"
                )
                == normalized
            ]

        return sorted(
            values,
            key=lambda item: (
                str(
                    item.get(
                        "image_path",
                        "",
                    )
                ).casefold()
            ),
        )

    def summary(
        self,
    ) -> dict[str, int]:
        records = self.records()

        correct = sum(
            1
            for item in records
            if item.get(
                "label"
            )
            == "correct"
        )

        wrong = sum(
            1
            for item in records
            if item.get(
                "label"
            )
            == "wrong"
        )

        skipped = sum(
            1
            for item in records
            if item.get(
                "label"
            )
            == "skip"
        )

        return {
            "total": len(
                records
            ),
            "correct": correct,
            "wrong": wrong,
            "skip": skipped,
            "trainable": (
                correct
                + wrong
            ),
        }

