from __future__ import annotations

import gc
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class SafeIOError(Exception):
    """Raised when a safe filesystem operation fails."""

    pass


_FILE_LOCK = threading.RLock()


def _wait(
    attempt: int,
    base_delay: float = 0.20,
    max_delay: float = 1.00,
) -> None:
    delay = min(
        max_delay,
        base_delay * max(1, attempt),
    )

    time.sleep(delay)


def _temporary_path(
    destination: Path,
) -> Path:
    return destination.with_name(
        destination.name
        + "."
        + uuid.uuid4().hex
        + ".tmp"
    )


def safe_write_bytes(
    destination: str | Path,
    content: bytes,
    retries: int = 30,
) -> Path:
    destination = Path(
        destination
    ).resolve()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = _temporary_path(
        destination
    )

    last_error = None

    with _FILE_LOCK:
        try:
            with temporary.open(
                "wb"
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(
                    handle.fileno()
                )

            for attempt in range(
                1,
                retries + 1,
            ):
                try:
                    os.replace(
                        temporary,
                        destination,
                    )

                    return destination

                except (
                    PermissionError,
                    OSError,
                ) as exc:
                    last_error = exc

                    gc.collect()

                    _wait(attempt)

            # ---------------------------------------------
            # FALLBACK
            # ---------------------------------------------

            for attempt in range(
                1,
                retries + 1,
            ):
                try:
                    with destination.open(
                        "wb"
                    ) as handle:
                        handle.write(content)
                        handle.flush()

                        os.fsync(
                            handle.fileno()
                        )

                    if temporary.exists():
                        temporary.unlink()

                    return destination

                except (
                    PermissionError,
                    OSError,
                ) as exc:
                    last_error = exc

                    gc.collect()

                    _wait(attempt)

            raise SafeIOError(
                "Could not safely write file after "
                f"{retries} retries: {destination}"
            ) from last_error

        finally:
            if temporary.exists():
                try:
                    temporary.unlink()

                except OSError:
                    pass


def safe_write_text(
    destination: str | Path,
    content: str,
    retries: int = 30,
    encoding: str = "utf-8",
) -> Path:
    return safe_write_bytes(
        destination=destination,
        content=content.encode(
            encoding
        ),
        retries=retries,
    )


def safe_write_json(
    destination: str | Path,
    payload: Any,
    retries: int = 30,
) -> Path:
    text = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )

    # Validate before writing.
    json.loads(text)

    result = safe_write_text(
        destination=destination,
        content=text,
        retries=retries,
    )

    # Validate committed file.
    try:
        json.loads(
            result.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        raise SafeIOError(
            "JSON verification failed after "
            f"write: {result}"
        ) from exc

    return result


def safe_copy_file(
    source: str | Path,
    destination: str | Path,
    retries: int = 40,
) -> Path:
    source = Path(
        source
    ).resolve()

    destination = Path(
        destination
    ).resolve()

    if not source.is_file():
        raise SafeIOError(
            "Source file does not exist: "
            f"{source}"
        )

    source_size = (
        source.stat().st_size
    )

    if source_size <= 0:
        raise SafeIOError(
            "Source file is empty: "
            f"{source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = _temporary_path(
        destination
    )

    last_error = None

    with _FILE_LOCK:
        try:
            for attempt in range(
                1,
                retries + 1,
            ):
                try:
                    shutil.copy2(
                        source,
                        temporary,
                    )

                    if (
                        temporary.stat().st_size
                        != source_size
                    ):
                        raise SafeIOError(
                            "Temporary copy size "
                            "verification failed."
                        )

                    break

                except (
                    PermissionError,
                    OSError,
                ) as exc:
                    last_error = exc

                    gc.collect()

                    _wait(attempt)

            else:
                raise SafeIOError(
                    "Could not stage file: "
                    f"{source}"
                ) from last_error

            for attempt in range(
                1,
                retries + 1,
            ):
                try:
                    os.replace(
                        temporary,
                        destination,
                    )

                    if (
                        destination.stat().st_size
                        != source_size
                    ):
                        raise SafeIOError(
                            "Destination copy size "
                            "verification failed."
                        )

                    return destination

                except (
                    PermissionError,
                    OSError,
                ) as exc:
                    last_error = exc

                    gc.collect()

                    _wait(attempt)

            # ---------------------------------------------
            # FALLBACK
            # ---------------------------------------------

            for attempt in range(
                1,
                retries + 1,
            ):
                try:
                    shutil.copy2(
                        source,
                        destination,
                    )

                    if (
                        destination.stat().st_size
                        != source_size
                    ):
                        raise SafeIOError(
                            "Fallback file size "
                            "verification failed."
                        )

                    return destination

                except (
                    PermissionError,
                    OSError,
                ) as exc:
                    last_error = exc

                    gc.collect()

                    _wait(attempt)

            raise SafeIOError(
                "Could not safely copy file after "
                f"{retries} retries: {destination}. "
                "A running Python/Streamlit process "
                "may still have the file locked."
            ) from last_error

        finally:
            if temporary.exists():
                try:
                    temporary.unlink()

                except OSError:
                    pass


def safe_remove_file(
    path: str | Path,
    retries: int = 20,
) -> None:
    target = Path(path)

    if not target.exists():
        return

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):
        try:
            target.unlink()

            return

        except FileNotFoundError:
            return

        except (
            PermissionError,
            OSError,
        ) as exc:
            last_error = exc

            gc.collect()

            _wait(attempt)

    raise SafeIOError(
        "Could not remove locked file: "
        f"{target}"
    ) from last_error


def safe_remove_tree(
    path: str | Path,
    retries: int = 20,
) -> None:
    target = Path(path)

    if not target.exists():
        return

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):
        try:
            shutil.rmtree(target)

            return

        except FileNotFoundError:
            return

        except (
            PermissionError,
            OSError,
        ) as exc:
            last_error = exc

            gc.collect()

            _wait(attempt)

    raise SafeIOError(
        "Could not remove locked directory: "
        f"{target}"
    ) from last_error
