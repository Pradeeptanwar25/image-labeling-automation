import json

import pytest
from PIL import Image

from src.dataset_scanner import DatasetScanner
from src.duplicate_manager import delete_root_duplicates
from src.image_renamer import build_rename_plan
from src.model_trainer import (
    ModelTrainingError,
    VersionedYOLOTrainer,
)


def _create_training_dataset(
    root,
    manifest_counts,
):
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\n",
        encoding="utf-8",
    )
    pretrained_model = root / "base.pt"
    pretrained_model.write_bytes(b"weights")

    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / f"{split}.jpg").write_bytes(b"image")
        (label_dir / f"{split}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest_counts),
        encoding="utf-8",
    )
    return dataset_yaml, pretrained_model


def _trainer(dataset_yaml, pretrained_model, model_root):
    return VersionedYOLOTrainer(
        selected_folder="S25",
        mapped_label="label",
        dataset_yaml=dataset_yaml,
        pretrained_model=pretrained_model,
        model_root=model_root,
    )


def test_model_trainer_rejects_stale_manifest(tmp_path):
    dataset_yaml, pretrained_model = _create_training_dataset(
        tmp_path,
        {
            "total_images": 232,
            "training_images": 186,
            "validation_images": 46,
        },
    )

    with pytest.raises(ModelTrainingError, match="manifest count"):
        _trainer(
            dataset_yaml,
            pretrained_model,
            tmp_path / "models",
        )


def test_model_trainer_accepts_matching_dataset_counts(tmp_path):
    dataset_yaml, pretrained_model = _create_training_dataset(
        tmp_path,
        {
            "total_images": 2,
            "training_images": 1,
            "validation_images": 1,
        },
    )

    trainer = _trainer(
        dataset_yaml,
        pretrained_model,
        tmp_path / "models",
    )

    assert trainer.versions_root.is_dir()


def test_model_trainer_rejects_unpaired_images_and_labels(tmp_path):
    dataset_yaml, pretrained_model = _create_training_dataset(
        tmp_path,
        {
            "total_images": 2,
            "training_images": 1,
            "validation_images": 1,
        },
    )
    (tmp_path / "labels" / "train" / "train.txt").rename(
        tmp_path / "labels" / "train" / "other.txt"
    )

    with pytest.raises(ModelTrainingError, match="images and labels"):
        _trainer(
            dataset_yaml,
            pretrained_model,
            tmp_path / "models",
        )


def test_rename_is_independent_of_sidecar_labels(tmp_path):
    folder = tmp_path / "S25"
    folder.mkdir()
    (folder / "source.jpg").write_bytes(b"image")
    (folder / "source.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    plan = build_rename_plan(folder)

    assert [
        item["new_name"]
        for item in plan
        if not item["unchanged"]
    ] == ["S25_000001.jpg"]


def test_rename_appends_new_images_after_standardized_names(
    tmp_path,
):
    folder = tmp_path / "S25"
    folder.mkdir()
    (folder / "S25_000001.jpg").write_bytes(b"labeled")
    (folder / "S25_000001.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (folder / "new.jpg").write_bytes(b"new")

    plan = build_rename_plan(folder)

    assert [
        item["new_name"]
        for item in plan
        if not item["unchanged"]
    ] == ["S25_000002.jpg"]


@pytest.mark.parametrize("changed_file", ["a.jpg", "b.jpg"])
def test_duplicate_deletion_skips_files_changed_since_scan(
    tmp_path,
    changed_file,
):
    original = Image.new("RGB", (10, 10), "white")
    original.save(tmp_path / "a.jpg")
    (tmp_path / "b.jpg").write_bytes(
        (tmp_path / "a.jpg").read_bytes()
    )
    report = DatasetScanner(tmp_path).scan()

    Image.new("RGB", (10, 10), "black").save(
        tmp_path / changed_file
    )

    result = delete_root_duplicates(report)

    assert result["deleted_count"] == 0
    assert result["failed_count"] >= 1
    assert (tmp_path / "a.jpg").is_file()
    assert (tmp_path / "b.jpg").is_file()
