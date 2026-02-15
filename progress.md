# DeepInfant Windows GUI + Training Progress Log

## Date
- 2026-02-15

## Goal
- Make DeepInfant easy to use on Windows with:
  - microphone recording path
  - file upload path
- Keep Python workflow on `uv`.
- Produce/verify a usable PyTorch checkpoint flow for local inference.

## What Was Implemented

### 1) Windows GUI
- Added `app.py` using Gradio with two explicit paths:
  - `Record` (microphone)
  - `Select File` (upload)
- Added prediction outputs:
  - predicted label
  - confidence
  - caregiver tip

### 2) Predictor Compatibility + Robustness
- Updated `predict.py` to:
  - load checkpoints more robustly (`state_dict`, `model_state_dict`, raw state dict)
  - infer class count from classifier weights
  - support both 5-class and 9-class label maps
  - fail clearly on missing/invalid checkpoint or unreadable audio

### 3) `uv` Project Setup and Instructions
- Added `pyproject.toml`.
- Added `AGENTS.md` guidance to use:
  - `uv add ...`
  - `uv run ...`
- Updated docs in `README.md` for GUI launch and training commands.
- Updated `requirements.txt` with GUI/runtime dependencies.

## What Went Wrong and How It Was Fixed

### Issue A: Training crashed immediately with transform error
- Root cause:
  - `DeepInfantDataset` accepted `transform=True/False` but later called it like a function.
- Fix:
  - treat boolean transform as augmentation toggle (`self.augment`)
  - only call `self.transform(...)` when it is callable

### Issue B: Training crashed with invalid reshape in model forward pass
- Error observed:
  - `RuntimeError: shape '[8, -1, 2560]' is invalid for input of size 2048`
- Root cause:
  - CNN ended with `AdaptiveAvgPool2d(1)` which destroyed temporal/frequency structure needed by LSTM reshape.
- Fix:
  - removed collapsing tail from CNN
  - preserved pooled frequency size to match `input_size=256*10` for the LSTM

### Issue C: Accuracy got stuck exactly at ~64.35% train and ~62.60% val
- Observation:
  - train/val accuracy matched majority class percentages exactly.
- Root cause:
  - severe class imbalance (`hungry` dominates dataset).
- Fix:
  - added imbalance handling in `train.py`:
    - `WeightedRandomSampler`
    - class-weighted `CrossEntropyLoss`
    - configurable via `--imbalance-mode {none,sampler,loss,both}` (default: `both`)

## Dataset Split Facts (from current processed metadata)
- Train: 474 samples
- Validation/Test: 123 samples
- Majority class was `hungry`:
  - train: `305/474 = 64.35%`
  - val: `77/123 = 62.60%`

## Current Recommended Commands

### Smoke test (3 epochs)
```powershell
uv run train.py --device cpu --epochs 3 --batch-size 32 --num-workers 0 --imbalance-mode both
```

### Full run
```powershell
uv run train.py --device cpu --epochs 50 --batch-size 32 --num-workers 0 --imbalance-mode both
```

### Launch GUI
```powershell
uv run app.py
```

## Artifacts to Keep Out of Git
- `processed_dataset/`
- `deepinfant.pth`
- local env/cache artifacts

## Notes for Future You
- If accuracy pins to a constant near majority proportion again, re-check:
  - class counts
  - sampler and loss weighting enabled
  - per-class metrics (not only accuracy)
