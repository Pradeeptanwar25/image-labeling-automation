from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel


class SimilarImageFinderError(Exception):
    """Raised when visually similar image search fails."""

    pass


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".jfif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


class SimilarImageFinder:
    """
    Find visually similar images using DINOv2 embeddings.

    The search supports one or multiple reference images.

    For every candidate image:
        1. Compare the candidate against every reference image.
        2. Select the highest similarity score.
        3. Keep the candidate when that best score passes the threshold.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name

        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        try:
            self.processor = (
                AutoImageProcessor.from_pretrained(
                    model_name
                )
            )

            self.model = (
                AutoModel.from_pretrained(
                    model_name
                )
                .to(self.device)
                .eval()
            )

        except Exception as exc:
            raise SimilarImageFinderError(
                "Could not load DINOv2 model: "
                f"{exc}"
            ) from exc

    @staticmethod
    def discover_images(
        search_root: str | Path,
    ) -> list[Path]:
        """
        Recursively discover supported image files.
        """

        root = (
            Path(search_root)
            .expanduser()
            .resolve()
        )

        if not root.is_dir():
            raise SimilarImageFinderError(
                f"Search folder not found: {root}"
            )

        return sorted(
            [
                path
                for path in root.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.casefold()
                    in SUPPORTED_EXTENSIONS
                )
            ],
            key=lambda path: (
                path.as_posix().casefold()
            ),
        )

    @staticmethod
    def normalize_reference_images(
        reference_images: (
            str
            | Path
            | Iterable[str | Path]
        ),
    ) -> list[Path]:
        """
        Convert one or multiple reference-image values
        into a validated, unique list of absolute paths.

        Backward compatibility:
            A single string or Path is still accepted.
        """

        if isinstance(
            reference_images,
            (str, Path),
        ):
            raw_values = [reference_images]
        else:
            try:
                raw_values = list(
                    reference_images
                )
            except TypeError as exc:
                raise SimilarImageFinderError(
                    "Reference images must be a path "
                    "or a list of paths."
                ) from exc

        if not raw_values:
            raise SimilarImageFinderError(
                "At least one reference image is required."
            )

        normalized_paths: list[Path] = []
        seen_paths: set[str] = set()

        for raw_value in raw_values:
            clean_value = (
                str(raw_value)
                .strip()
                .strip('"')
                .strip("'")
            )

            if not clean_value:
                continue

            path = (
                Path(clean_value)
                .expanduser()
                .resolve()
            )

            path_key = str(path).casefold()

            if path_key in seen_paths:
                continue

            if not path.is_file():
                raise SimilarImageFinderError(
                    "Reference image not found: "
                    f"{path}"
                )

            if (
                path.suffix.casefold()
                not in SUPPORTED_EXTENSIONS
            ):
                raise SimilarImageFinderError(
                    "Unsupported reference image type: "
                    f"{path}"
                )

            seen_paths.add(path_key)
            normalized_paths.append(path)

        if not normalized_paths:
            raise SimilarImageFinderError(
                "At least one valid reference image "
                "is required."
            )

        return normalized_paths

    @staticmethod
    def calculate_blur_score(
        image_path: str | Path,
    ) -> float:
        """
        Measure sharpness with Variance of Laplacian.

        A lower score usually means a blurrier image.
        This is an image-quality check, not an object detector.
        """

        path = (
            Path(image_path)
            .expanduser()
            .resolve()
        )

        if not path.is_file():
            raise SimilarImageFinderError(
                f"Image not found: {path}"
            )

        try:
            with Image.open(path) as source:
                grayscale_image = (
                    ImageOps.exif_transpose(source)
                    .convert("L")
                )

                grayscale_array = np.asarray(
                    grayscale_image,
                    dtype=np.uint8,
                )

            if grayscale_array.size == 0:
                raise SimilarImageFinderError(
                    f"Image has no readable pixels: {path}"
                )

            return float(
                cv2.Laplacian(
                    grayscale_array,
                    cv2.CV_64F,
                ).var()
            )

        except SimilarImageFinderError:
            raise

        except Exception as exc:
            raise SimilarImageFinderError(
                "Could not calculate blur score "
                f"for {path}: {exc}"
            ) from exc

    def create_embedding(
        self,
        image_path: str | Path,
    ) -> torch.Tensor:
        """
        Create a normalized DINOv2 embedding.

        Flow:
            PIL open
            -> EXIF orientation correction
            -> RGB conversion
            -> DINOv2 processor
            -> DINOv2 model
            -> CLS/global embedding
            -> L2 normalization
        """

        path = (
            Path(image_path)
            .expanduser()
            .resolve()
        )

        if not path.is_file():
            raise SimilarImageFinderError(
                f"Image not found: {path}"
            )

        try:
            with Image.open(path) as source:
                image = (
                    ImageOps.exif_transpose(
                        source
                    )
                    .convert("RGB")
                )

                inputs = self.processor(
                    images=image,
                    return_tensors="pt",
                )

            inputs = {
                key: value.to(self.device)
                for key, value
                in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model(
                    **inputs
                )

                embedding = (
                    outputs.last_hidden_state[
                        :,
                        0,
                        :,
                    ]
                )

                embedding = F.normalize(
                    embedding,
                    p=2,
                    dim=1,
                )

            return embedding.cpu()

        except SimilarImageFinderError:
            raise

        except Exception as exc:
            raise SimilarImageFinderError(
                f"Could not embed image {path}: "
                f"{exc}"
            ) from exc

    def create_reference_embeddings(
        self,
        reference_paths: list[Path],
    ) -> list[dict[str, Any]]:
        """
        Create an embedding for every reference image.
        """

        reference_records: list[
            dict[str, Any]
        ] = []

        for reference_path in reference_paths:
            embedding = self.create_embedding(
                reference_path
            )

            reference_records.append(
                {
                    "path": reference_path,
                    "embedding": embedding,
                }
            )

        return reference_records

    def search(
        self,
        reference_images: (
            str
            | Path
            | Iterable[str | Path]
            | None
        ) = None,
        search_root: str | Path | None = None,
        minimum_similarity: float = 0.80,
        maximum_results: int = 200,
        exclude_blurry_images: bool = True,
        blur_threshold: float = 100.0,
        negative_reference_images: (
            str
            | Path
            | Iterable[str | Path]
            | None
        ) = None,
        exclude_negative_matches: bool = False,
        negative_similarity_threshold: float = 0.75,
        negative_similarity_margin: float = 0.03,
        positive_scoring_mode: str = "high_recall",
        positive_agreement_threshold: float = 0.45,
        minimum_reference_agreement: int = 1,
        reference_image: (
            str | Path | None
        ) = None,
    ) -> dict[str, Any]:
        """
        Search for visually similar images.

        `reference_image` is retained for compatibility
        with the old single-reference caller.

        Final candidate score:
            highest cosine similarity obtained against
            any supplied reference image.
        """

        if reference_images is None:
            reference_images = reference_image

        if reference_images is None:
            raise SimilarImageFinderError(
                "At least one reference image is required."
            )

        if search_root is None:
            raise SimilarImageFinderError(
                "Search root folder is required."
            )

        reference_paths = (
            self.normalize_reference_images(
                reference_images
            )
        )

        root = (
            Path(search_root)
            .expanduser()
            .resolve()
        )

        if not root.is_dir():
            raise SimilarImageFinderError(
                f"Search folder not found: {root}"
            )

        threshold = float(
            minimum_similarity
        )

        if not 0.0 <= threshold <= 1.0:
            raise SimilarImageFinderError(
                "Similarity threshold must "
                "be between 0 and 1."
            )

        try:
            result_limit = int(
                maximum_results
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise SimilarImageFinderError(
                "Maximum results must be an integer."
            ) from exc

        if result_limit < 1:
            raise SimilarImageFinderError(
                "Maximum results must be at least 1."
            )

        try:
            validated_blur_threshold = float(
                blur_threshold
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise SimilarImageFinderError(
                "Blur threshold must be a number."
            ) from exc

        if validated_blur_threshold < 0:
            raise SimilarImageFinderError(
                "Blur threshold cannot be negative."
            )

        try:
            validated_negative_threshold = float(
                negative_similarity_threshold
            )
            validated_negative_margin = float(
                negative_similarity_margin
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise SimilarImageFinderError(
                "Negative similarity threshold and "
                "margin must be numbers."
            ) from exc

        if not (
            0.0
            <= validated_negative_threshold
            <= 1.0
        ):
            raise SimilarImageFinderError(
                "Negative similarity threshold must "
                "be between 0 and 1."
            )

        if not (
            0.0
            <= validated_negative_margin
            <= 1.0
        ):
            raise SimilarImageFinderError(
                "Negative similarity margin must "
                "be between 0 and 1."
            )

        validated_scoring_mode = (
            str(positive_scoring_mode)
            .strip()
            .casefold()
        )

        supported_scoring_modes = {
            "high_recall",
            "balanced",
            "strict",
        }

        if (
            validated_scoring_mode
            not in supported_scoring_modes
        ):
            raise SimilarImageFinderError(
                "Positive scoring mode must be one of: "
                "high_recall, balanced, strict."
            )

        try:
            validated_agreement_threshold = float(
                positive_agreement_threshold
            )
            validated_minimum_agreement = int(
                minimum_reference_agreement
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise SimilarImageFinderError(
                "Positive agreement threshold must be "
                "a number and minimum reference "
                "agreement must be an integer."
            ) from exc

        if not (
            0.0
            <= validated_agreement_threshold
            <= 1.0
        ):
            raise SimilarImageFinderError(
                "Positive agreement threshold must "
                "be between 0 and 1."
            )

        if validated_minimum_agreement < 1:
            raise SimilarImageFinderError(
                "Minimum reference agreement must "
                "be at least 1."
            )

        reference_count = len(
            reference_paths
        )

        effective_minimum_agreement = min(
            validated_minimum_agreement,
            reference_count,
        )

        reference_records = (
            self.create_reference_embeddings(
                reference_paths
            )
        )

        negative_reference_paths: list[Path] = []
        negative_reference_records: list[
            dict[str, Any]
        ] = []

        if (
            exclude_negative_matches
            and negative_reference_images
            is not None
        ):
            negative_reference_paths = (
                self.normalize_reference_images(
                    negative_reference_images
                )
            )

            positive_path_keys = {
                str(path).casefold()
                for path in reference_paths
            }

            overlap_paths = [
                path
                for path in negative_reference_paths
                if str(path).casefold()
                in positive_path_keys
            ]

            if overlap_paths:
                raise SimilarImageFinderError(
                    "The same image cannot be both "
                    "a positive and negative reference: "
                    f"{overlap_paths[0]}"
                )

            negative_reference_records = (
                self.create_reference_embeddings(
                    negative_reference_paths
                )
            )

        reference_path_keys = {
            str(path).casefold()
            for path in reference_paths
        }

        negative_reference_path_keys = {
            str(path).casefold()
            for path in negative_reference_paths
        }

        all_reference_path_keys = (
            reference_path_keys
            | negative_reference_path_keys
        )

        candidates = self.discover_images(
            root
        )

        matches: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        blurry_images: list[dict[str, Any]] = []
        negative_excluded: list[
            dict[str, Any]
        ] = []

        candidate_count = 0
        dinov2_compared_count = 0

        for candidate in candidates:
            candidate_key = str(
                candidate.resolve()
            ).casefold()

            if candidate_key in all_reference_path_keys:
                continue

            candidate_count += 1

            try:
                blur_score = self.calculate_blur_score(
                    candidate
                )

                if (
                    exclude_blurry_images
                    and blur_score
                    < validated_blur_threshold
                ):
                    blurry_images.append(
                        {
                            "image_path": str(
                                candidate
                            ),
                            "relative_path": (
                                candidate
                                .relative_to(root)
                                .as_posix()
                            ),
                            "blur_score": round(
                                blur_score,
                                2,
                            ),
                            "reason": "Blurry image",
                        }
                    )
                    continue

                candidate_embedding = (
                    self.create_embedding(
                        candidate
                    )
                )

                dinov2_compared_count += 1

                reference_scores: list[
                    dict[str, Any]
                ] = []

                for reference_record in (
                    reference_records
                ):
                    similarity = float(
                        F.cosine_similarity(
                            reference_record[
                                "embedding"
                            ],
                            candidate_embedding,
                        ).item()
                    )

                    reference_scores.append(
                        {
                            "reference_path": str(
                                reference_record[
                                    "path"
                                ]
                            ),
                            "reference_name": (
                                reference_record[
                                    "path"
                                ].name
                            ),
                            "similarity": similarity,
                            "similarity_percent": round(
                                similarity * 100,
                                2,
                            ),
                        }
                    )

                best_reference = max(
                    reference_scores,
                    key=lambda item: (
                        item["similarity"]
                    ),
                )

                best_similarity = float(
                    best_reference[
                        "similarity"
                    ]
                )

                positive_similarity_values = sorted(
                    [
                        float(item["similarity"])
                        for item in reference_scores
                    ],
                    reverse=True,
                )

                top_two_values = (
                    positive_similarity_values[:2]
                )

                top_two_average = (
                    sum(top_two_values)
                    / len(top_two_values)
                )

                all_reference_average = (
                    sum(positive_similarity_values)
                    / len(positive_similarity_values)
                )

                sorted_for_median = sorted(
                    positive_similarity_values
                )

                median_index = (
                    len(sorted_for_median)
                    // 2
                )

                if (
                    len(sorted_for_median)
                    % 2
                    == 1
                ):
                    median_similarity = (
                        sorted_for_median[
                            median_index
                        ]
                    )
                else:
                    median_similarity = (
                        sorted_for_median[
                            median_index - 1
                        ]
                        + sorted_for_median[
                            median_index
                        ]
                    ) / 2.0

                references_agreed = sum(
                    1
                    for value
                    in positive_similarity_values
                    if value
                    >= validated_agreement_threshold
                )

                agreement_ratio = (
                    references_agreed
                    / len(positive_similarity_values)
                )

                if (
                    validated_scoring_mode
                    == "high_recall"
                ):
                    combined_similarity = (
                        0.70 * best_similarity
                        + 0.20 * top_two_average
                        + 0.10 * all_reference_average
                    )

                    agreement_passed = (
                        references_agreed
                        >= effective_minimum_agreement
                    )

                elif (
                    validated_scoring_mode
                    == "balanced"
                ):
                    combined_similarity = (
                        0.50 * best_similarity
                        + 0.30 * top_two_average
                        + 0.20 * all_reference_average
                    )

                    required_agreement = min(
                        max(
                            2,
                            effective_minimum_agreement,
                        ),
                        len(
                            positive_similarity_values
                        ),
                    )

                    agreement_passed = (
                        references_agreed
                        >= required_agreement
                    )

                else:
                    combined_similarity = (
                        0.35 * best_similarity
                        + 0.35 * top_two_average
                        + 0.30 * all_reference_average
                    )

                    required_agreement = min(
                        max(
                            3,
                            effective_minimum_agreement,
                        ),
                        len(
                            positive_similarity_values
                        ),
                    )

                    agreement_passed = (
                        references_agreed
                        >= required_agreement
                    )

                final_similarity = float(
                    combined_similarity
                )

                negative_reference_scores: list[
                    dict[str, Any]
                ] = []

                for negative_record in (
                    negative_reference_records
                ):
                    negative_similarity = float(
                        F.cosine_similarity(
                            negative_record[
                                "embedding"
                            ],
                            candidate_embedding,
                        ).item()
                    )

                    negative_reference_scores.append(
                        {
                            "reference_path": str(
                                negative_record[
                                    "path"
                                ]
                            ),
                            "reference_name": (
                                negative_record[
                                    "path"
                                ].name
                            ),
                            "similarity": (
                                negative_similarity
                            ),
                            "similarity_percent": round(
                                negative_similarity
                                * 100,
                                2,
                            ),
                        }
                    )

                best_negative_reference = None
                best_negative_similarity = None
                negative_difference = None
                rejected_by_negative = False

                if negative_reference_scores:
                    best_negative_reference = max(
                        negative_reference_scores,
                        key=lambda item: (
                            item["similarity"]
                        ),
                    )

                    best_negative_similarity = float(
                        best_negative_reference[
                            "similarity"
                        ]
                    )

                    negative_difference = (
                        best_negative_similarity
                        - final_similarity
                    )

                    rejected_by_negative = (
                        best_negative_similarity
                        >= validated_negative_threshold
                        and negative_difference
                        >= validated_negative_margin
                    )

                if rejected_by_negative:
                    negative_excluded.append(
                        {
                            "image_path": str(
                                candidate
                            ),
                            "relative_path": (
                                candidate
                                .relative_to(root)
                                .as_posix()
                            ),
                            "blur_score": round(
                                blur_score,
                                2,
                            ),
                            "best_positive_similarity": (
                                best_similarity
                            ),
                            "best_positive_percent": round(
                                best_similarity * 100,
                                2,
                            ),
                            "best_positive_reference": (
                                best_reference[
                                    "reference_name"
                                ]
                            ),
                            "final_positive_similarity": (
                                final_similarity
                            ),
                            "final_positive_percent": round(
                                final_similarity * 100,
                                2,
                            ),
                            "top_two_average": (
                                top_two_average
                            ),
                            "all_reference_average": (
                                all_reference_average
                            ),
                            "median_similarity": (
                                median_similarity
                            ),
                            "references_agreed": (
                                references_agreed
                            ),
                            "reference_count": (
                                len(
                                    positive_similarity_values
                                )
                            ),
                            "agreement_ratio": (
                                agreement_ratio
                            ),
                            "best_negative_similarity": (
                                best_negative_similarity
                            ),
                            "best_negative_percent": round(
                                best_negative_similarity
                                * 100,
                                2,
                            ),
                            "best_negative_reference": (
                                best_negative_reference[
                                    "reference_name"
                                ]
                            ),
                            "negative_difference": round(
                                negative_difference,
                                4,
                            ),
                            "reason": (
                                "Closer to excluded "
                                "negative reference"
                            ),
                            "negative_reference_scores": (
                                sorted(
                                    negative_reference_scores,
                                    key=lambda item: (
                                        -item[
                                            "similarity"
                                        ]
                                    ),
                                )
                            ),
                        }
                    )
                    continue

                if (
                    final_similarity >= threshold
                    and agreement_passed
                ):
                    matches.append(
                        {
                            "image_path": str(
                                candidate
                            ),
                            "relative_path": (
                                candidate
                                .relative_to(root)
                                .as_posix()
                            ),
                            "similarity": (
                                final_similarity
                            ),
                            "final_similarity": (
                                final_similarity
                            ),
                            "final_similarity_percent": (
                                round(
                                    final_similarity
                                    * 100,
                                    2,
                                )
                            ),
                            "best_positive_similarity": (
                                best_similarity
                            ),
                            "best_positive_percent": (
                                round(
                                    best_similarity
                                    * 100,
                                    2,
                                )
                            ),
                            "top_two_average": (
                                top_two_average
                            ),
                            "top_two_average_percent": (
                                round(
                                    top_two_average
                                    * 100,
                                    2,
                                )
                            ),
                            "all_reference_average": (
                                all_reference_average
                            ),
                            "all_reference_average_percent": (
                                round(
                                    all_reference_average
                                    * 100,
                                    2,
                                )
                            ),
                            "median_similarity": (
                                median_similarity
                            ),
                            "median_similarity_percent": (
                                round(
                                    median_similarity
                                    * 100,
                                    2,
                                )
                            ),
                            "references_agreed": (
                                references_agreed
                            ),
                            "reference_count": (
                                len(
                                    positive_similarity_values
                                )
                            ),
                            "agreement_ratio": (
                                agreement_ratio
                            ),
                            "agreement_percent": (
                                round(
                                    agreement_ratio
                                    * 100,
                                    2,
                                )
                            ),
                            "scoring_mode": (
                                validated_scoring_mode
                            ),
                            "agreement_threshold": (
                                validated_agreement_threshold
                            ),
                            "blur_score": round(
                                blur_score,
                                2,
                            ),
                            "best_negative_similarity": (
                                best_negative_similarity
                            ),
                            "best_negative_percent": (
                                round(
                                    best_negative_similarity
                                    * 100,
                                    2,
                                )
                                if best_negative_similarity
                                is not None
                                else None
                            ),
                            "best_negative_reference": (
                                best_negative_reference[
                                    "reference_name"
                                ]
                                if best_negative_reference
                                is not None
                                else None
                            ),
                            "similarity_percent": (
                                round(
                                    final_similarity
                                    * 100,
                                    2,
                                )
                            ),
                            "best_reference_path": (
                                best_reference[
                                    "reference_path"
                                ]
                            ),
                            "best_reference_name": (
                                best_reference[
                                    "reference_name"
                                ]
                            ),
                            "reference_scores": sorted(
                                reference_scores,
                                key=lambda item: (
                                    -item[
                                        "similarity"
                                    ],
                                    item[
                                        "reference_name"
                                    ].casefold(),
                                ),
                            ),
                        }
                    )

            except Exception as exc:
                failures.append(
                    {
                        "image_path": str(
                            candidate
                        ),
                        "error": str(exc),
                    }
                )

        matches.sort(
            key=lambda item: (
                -item["similarity"],
                item[
                    "relative_path"
                ].casefold(),
            )
        )

        matches = matches[
            :result_limit
        ]

        reference_image_strings = [
            str(path)
            for path in reference_paths
        ]

        return {
            "status": "completed",
            "model_name": self.model_name,
            "device": str(self.device),

            # Old compatibility field:
            "reference_image": (
                reference_image_strings[0]
            ),

            # New multi-reference fields:
            "reference_images": (
                reference_image_strings
            ),
            "reference_count": len(
                reference_image_strings
            ),

            "search_root": str(root),
            "minimum_similarity": threshold,
            "positive_scoring_mode": (
                validated_scoring_mode
            ),
            "positive_agreement_threshold": (
                validated_agreement_threshold
            ),
            "minimum_reference_agreement": (
                effective_minimum_agreement
            ),
            "discovered_images": len(
                candidates
            ),
            "images_scanned": candidate_count,
            "matches_found": len(matches),
            "failed_count": len(failures),
            "exclude_blurry_images": bool(
                exclude_blurry_images
            ),
            "blur_threshold": (
                validated_blur_threshold
            ),
            "blurry_skipped": len(
                blurry_images
            ),
            "blurry_images": blurry_images,
            "dinov2_compared": (
                dinov2_compared_count
            ),
            "exclude_negative_matches": bool(
                exclude_negative_matches
            ),
            "negative_reference_images": [
                str(path)
                for path in negative_reference_paths
            ],
            "negative_reference_count": len(
                negative_reference_paths
            ),
            "negative_similarity_threshold": (
                validated_negative_threshold
            ),
            "negative_similarity_margin": (
                validated_negative_margin
            ),
            "negative_excluded_count": len(
                negative_excluded
            ),
            "negative_excluded": (
                negative_excluded
            ),
            "matches": matches,
            "failures": failures,
            "scoring_method": (
                "combined multi-reference positive "
                "similarity with agreement checking "
                "and optional negative-reference "
                "exclusion"
            ),
        }


def copy_similarity_matches(
    search_result: dict[str, Any],
    output_folder: str | Path,
) -> dict[str, Any]:
    """
    Copy matched images to a separate output folder.

    Original images are not modified.
    """

    output_root = (
        Path(output_folder)
        .expanduser()
        .resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for index, match in enumerate(
        search_result.get(
            "matches",
            [],
        ),
        start=1,
    ):
        source = Path(
            match[
                "image_path"
            ]
        )

        if not source.is_file():
            failed.append(
                {
                    "source": str(source),
                    "error": (
                        "Source image missing."
                    ),
                }
            )
            continue

        short_hash = hashlib.sha256(
            str(source).encode(
                "utf-8"
            )
        ).hexdigest()[:10]

        destination = (
            output_root
            / (
                f"{index:04d}_"
                f"{short_hash}_"
                f"{source.name}"
            )
        )

        try:
            shutil.copy2(
                source,
                destination,
            )

            copied.append(
                {
                    "source": str(source),
                    "destination": str(
                        destination
                    ),
                    "similarity": (
                        match[
                            "similarity"
                        ]
                    ),
                    "best_reference_path": (
                        match.get(
                            "best_reference_path"
                        )
                    ),
                }
            )

        except OSError as exc:
            failed.append(
                {
                    "source": str(source),
                    "error": str(exc),
                }
            )

    report = {
        "status": (
            "completed"
            if not failed
            else "completed_with_errors"
        ),
        "output_folder": str(
            output_root
        ),
        "reference_images": (
            search_result.get(
                "reference_images",
                [
                    search_result.get(
                        "reference_image"
                    )
                ],
            )
        ),
        "copied_count": len(
            copied
        ),
        "failed_count": len(
            failed
        ),
        "copied_files": copied,
        "failed_files": failed,
    }

    report_path = (
        output_root
        / "similarity_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report[
        "report_file"
    ] = str(
        report_path
    )

    return report