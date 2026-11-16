# Ericsson Image Labelling Automation - Milestone 1

Implemented now: recursive dataset scan, image metadata validation, corrupted-image detection, SHA-256 duplicate detection, master-label parsing, explicit folder-to-label mapping, Streamlit dashboard, JSON report, and tests.

## Architecture decisions
- Original images are read-only. The scanner never modifies them.
- Folder labels are explicit in `config/label_mapping.json`; folder names are never treated as labels automatically.
- Image paths are stored relative to the dataset root for portability.
- SHA-256 identifies byte-identical duplicates. Near-duplicate/perceptual detection is planned separately.
- Milestone 1 stays model-independent. Later milestones can replace the detector without changing label/scanner contracts.

## Required pilot inputs
1. Dataset root such as `C:/Users/304758/Downloads/classification3`.
2. The actual `Ericson_classification label.txt` in that root.
3. S25 images in `classification3/S25/`.

## Windows setup
Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Choose a dataset
On app startup, enter the dataset root in the sidebar. The included mapping starts with:

```json
{"S25": "s25_full_tower_view"}
```

The dashboard validates that this label exists in the actual master label file.

Command-line annotation, training, and inference tools require the dataset root explicitly:

```powershell
python -m src.training_dataset --folder "S26/S26_3" --dataset-root "C:\path\to\dataset"
python -m src.model_trainer --folder "S26/S26_3" --dataset-root "C:\path\to\dataset"
python -m src.annotation_importer --folder "S26/S26_3" --dataset-root "C:\path\to\dataset"
python -m src.annotation_merger --folder "S26/S26_3" --dataset-root "C:\path\to\dataset"
python -m src.inference --folder "S26/S26_3" --dataset-root "C:\path\to\dataset"
```

## Acceptance check
- Scan completes without modifying images.
- S25 count appears.
- Corrupt files and exact duplicate groups are visible.
- Every folder is mapped manually to a valid master label.
- `pytest -q` passes.

## Next milestone contract
Milestone 2 will import actual Make Sense AI exports and normalize them to pixel-space `xmin,ymin,xmax,ymax`. Before implementation, retain one real S25 export of each required format (YOLO ZIP, VOC ZIP, CSV) so directory layout and coordinate conventions can be tested against the exact exporter output.
