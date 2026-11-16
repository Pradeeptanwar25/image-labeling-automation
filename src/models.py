from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ImageRecord:
    path: str
    relative_path: str
    folder: str
    width: int | None
    height: int | None
    channels: int | None
    size_bytes: int
    sha256: str | None
    status: str
    error: str | None = None

    def to_dict(self):
        return asdict(self)
