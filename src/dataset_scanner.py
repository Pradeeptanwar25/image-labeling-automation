from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .models import ImageRecord


class DatasetScanner:
    """
    Scan a dataset recursively.

    Features:
    - Detect supported image files
    - Validate images using Pillow
    - Read image width, height and channels
    - Generate SHA-256 hashes
    - Detect exact duplicate files
    - Count images by nested category folder
    - Preserve relative image paths
    """

    def __init__(
        self,
        root,
        extensions=None,
    ):
        """
        Initialize the dataset scanner.

        Parameters
        ----------
        root:
            Root directory containing the dataset.

        extensions:
            Optional list of supported image extensions.
        """

        self.root = Path(root).resolve()

        default_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        ]

        selected_extensions = (
            extensions
            if extensions
            else default_extensions
        )

        self.extensions = {
            extension.lower()
            for extension in selected_extensions
        }

    @staticmethod
    def _hash(path):
        """
        Generate a SHA-256 fingerprint from
        the raw bytes of an image file.
        """

        hash_object = sha256()

        with path.open("rb") as image_file:

            for chunk in iter(
                lambda: image_file.read(
                    1024 * 1024
                ),
                b"",
            ):
                hash_object.update(chunk)

        return hash_object.hexdigest()

    def _get_category_folder(
        self,
        relative_path,
    ):
        """
        Return the image's containing folder relative
        to the dataset root.

        Examples
        --------
        S16/S16_1/image001.jpg
            -> S16/S16_1

        S26_2/image001.jpg
            -> S26_2

        image001.jpg
            -> .
        """

        parent_folder = relative_path.parent

        if str(parent_folder) == ".":
            return "."

        return parent_folder.as_posix()

    def scan(self):
        """
        Scan the complete dataset and return
        a normalized scan report.
        """

        if not self.root.is_dir():

            raise NotADirectoryError(
                "Dataset root not found: "
                f"{self.root}"
            )

        records = []

        # -------------------------------------------------
        # Find supported images recursively
        # -------------------------------------------------

        for image_path in sorted(
            self.root.rglob("*")
        ):

            if not image_path.is_file():
                continue

            if (
                image_path.suffix.lower()
                not in self.extensions
            ):
                continue

            relative_path = (
                image_path.relative_to(
                    self.root
                )
            )

            category_folder = (
                self._get_category_folder(
                    relative_path
                )
            )

            file_size = (
                image_path.stat().st_size
            )

            try:

                # -----------------------------------------
                # First pass:
                # Verify that Pillow can decode the image.
                # -----------------------------------------

                with Image.open(
                    image_path
                ) as image:

                    image.verify()

                # -----------------------------------------
                # Second pass:
                # Read image metadata.
                # Image must be reopened after verify().
                # -----------------------------------------

                with Image.open(
                    image_path
                ) as image:

                    width, height = (
                        image.size
                    )

                    channels = len(
                        image.getbands()
                    )

                # -----------------------------------------
                # SHA-256 exact duplicate fingerprint
                # -----------------------------------------

                file_hash = self._hash(
                    image_path
                )

                record = ImageRecord(
                    path=str(
                        image_path
                    ),
                    relative_path=(
                        relative_path.as_posix()
                    ),
                    folder=category_folder,
                    width=width,
                    height=height,
                    channels=channels,
                    size_bytes=file_size,
                    sha256=file_hash,
                    status="ok",
                    error=None,
                )

                records.append(record)

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as exception:

                # -----------------------------------------
                # Unreadable or damaged image
                # -----------------------------------------

                record = ImageRecord(
                    path=str(
                        image_path
                    ),
                    relative_path=(
                        relative_path.as_posix()
                    ),
                    folder=category_folder,
                    width=None,
                    height=None,
                    channels=None,
                    size_bytes=file_size,
                    sha256=None,
                    status="corrupt",
                    error=str(exception),
                )

                records.append(record)

        # -------------------------------------------------
        # Group valid files by SHA-256
        # -------------------------------------------------

        hash_groups = defaultdict(list)

        for record in records:

            if record.sha256:

                hash_groups[
                    record.sha256
                ].append(
                    record.relative_path
                )

        # Keep only hashes associated
        # with two or more files.
        duplicates = {

            file_hash: image_paths

            for (
                file_hash,
                image_paths,
            ) in hash_groups.items()

            if len(image_paths) > 1
        }

        # -------------------------------------------------
        # Count images by actual containing folder
        # -------------------------------------------------

        folder_counts = defaultdict(int)

        for record in records:

            folder_counts[
                record.folder
            ] += 1

        # -------------------------------------------------
        # Prepare final report
        # -------------------------------------------------

        report = {

            "root": str(
                self.root
            ),

            "records": [
                record.to_dict()
                for record in records
            ],

            "folder_counts": dict(
                sorted(
                    folder_counts.items()
                )
            ),

            "duplicates": duplicates,

            "corrupt_count": sum(
                record.status == "corrupt"
                for record in records
            ),
        }

        return report