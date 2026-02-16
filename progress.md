# DeepInfant Progress Log

## Date Range
- 2026-02-15 to 2026-02-16

## Project Goals
- Keep local workflow simple on Windows with `uv`.
- Train a usable PyTorch checkpoint for local inference and GUI use.
- Improve model behavior on imbalanced 9-class data.
- Make training/cross-validation logs easier to read and share.

## Work Completed

### 1) Windows GUI + Inference Flow
- Added `app.py` (Gradio) with:
  - `Record` path (microphone)
  - `Select File` path (upload)
- Updated `predict.py` for robust checkpoint loading:
  - supports raw state dict / `state_dict` / `model_state_dict`
  - infers class count from classifier shape
  - supports both 5-class and 9-class labels
  - clearer failure messages for bad checkpoints/audio

### 2) Environment + Packaging
- Added/updated project metadata for `uv` workflow.
- Documented usage in `README.md`.
- Standardized commands around:
  - `uv add ...`
  - `uv run ...`

### 3) Training Pipeline Fixes (Stability + Correctness)
- Fixed dataset transform/augmentation handling bug:
  - bool transform is treated as augmentation toggle
  - callable transforms are invoked only when callable
- Fixed model shape flow:
  - preserved temporal/frequency structure needed by LSTM input
  - removed incompatible collapsing behavior that caused reshape failure

### 4) Imbalance and Metric Improvements
- Added configurable imbalance handling:
  - `--imbalance-mode {none,sampler,loss,both}`
  - class-weighting schemes with clipping
- Switched monitoring emphasis to macro-quality metrics:
  - macro-F1
  - balanced accuracy

### 5) Dataset Preparation Improvements
- `prepare_dataset.py` updated to support:
  - `--unknown-strategy {keep,cap,drop}`
  - `--unknown-cap-ratio` (default `0.35`)
  - stratified split option (`--stratify`, default `true`)
  - minimum per-class guard (`--min-per-class`, default `5`)
  - broader audio extensions support (`.wav`, `.mp3`, `.caf`, `.3gp`, `.ogg`, `.m4a`)
- Added cleanup of stale generated WAVs before re-preparing split.
- Fixed edge case: `unknown=drop` no longer fails min-class guard for unknown.

### 6) Training Strategy Upgrade (`train.py`)
- Added/changed defaults focused on stable generalization:
  - `--epochs 60`
  - `--lr 3e-4`
  - `--weight-decay 5e-4`
  - `--imbalance-mode loss`
  - `--scheduler cosine`
  - `--max-class-weight 3.0`
  - `--label-smoothing 0.05`
- Added:
  - monitor selection (`--monitor val_macro_f1|val_loss`)
  - early stopping (`--early-stop-patience`)
  - focal-loss option (`--focal-gamma`)
  - mixup option (`--mixup-alpha`)
  - multi-seed runs (`--seed-list`)
- Kept checkpointing based on selected monitor and printout of key metrics each epoch.

### 7) Cross-Validation Entry Point (`crossval.py`)
- Added stratified K-fold training runner:
  - `--cv-folds` (default `5`)
  - same optimizer/loss/scheduler knobs as `train.py`
  - same unknown handling and min-class guards
- Outputs fold metrics + aggregate mean/std to CSV (`--output-csv`).

### 8) Log Noise Cleanup (Crossval Paste-Friendly Output)
- Added targeted warning filtering (default ON) to suppress recurring noise only:
  - `PySoundFile failed. Trying audioread instead`
  - librosa `__audioread_load` deprecation warning
- Added warning controls:
  - `--show-warnings` (default `false`)
  - `--warning-log-path` (optional file for suppressed warnings)
- Added progress-bar controls:
  - `train.py`: `--no-progress` default `false`
  - `crossval.py`: `--no-progress` default `true` (cleaner shared logs)

## Latest Training Result Snapshot (Single Seed)
- Device: `cuda`
- Seed: `42`
- Train/Val samples: `281 / 71`
- Best validation metrics observed:
  - `Val Macro-F1: 0.6466`
  - `Val Accuracy: 67.61%`
  - `Val Balanced Accuracy: 0.6201`
- Early stopping triggered at epoch `59` (patience `10`).

## Current Recommended Commands

### Prepare data (capped unknown, stratified split)
```powershell
uv run prepare_dataset.py --unknown-strategy cap --unknown-cap-ratio 0.35 --stratify true --min-per-class 5
```

### Single training run (default improved settings)
```powershell
uv run train.py --device cuda
```

### Multi-seed training
```powershell
uv run train.py --device cuda --seed-list 42,43
```

### Cross-validation (clean logs by default)
```powershell
uv run crossval.py --device cuda --cv-folds 5 --output-csv cv_results.csv
```

### Cross-validation with warning visibility
```powershell
uv run crossval.py --device cuda --cv-folds 5 --show-warnings true
```

## Notes / Known Constraints
- Dataset is still relatively small and class-imbalanced; fold variance is expected.
- Current model is improved and usable, but further gains likely require:
  - more/cleaner data for minority classes
  - repeated-seed CV comparisons
  - targeted augmentation tuning

## Artifacts to Keep Out of Git
- `processed_dataset/`
- `deepinfant.pth`
- `deepinfant_cv_fold*.pth`
- local cache and environment artifacts
