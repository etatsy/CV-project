# CIFAR-10 Image Classification: Baseline → CIFAR-friendly models + MixUp/CutMix

This project builds a **reproducible experiment pipeline** for CIFAR‑10 image classification and compares several modeling choices.

## Project narrative (what I did)

1) **Start with a baseline**: transfer learning with ResNet50 + custom head (2‑stage training: freeze → fine‑tune).  
2) **Scale data and augmentations**: tried 10k → 20k → 50k and basic augmentation, but observed a **generalization gap** (train noticeably higher than val/test).  
3) **Change direction**: switch to CIFAR‑friendly scratch models (ResNet18 / WideResNet) and add regularization (MixUp / CutMix).  
4) **Log everything** into a single table for comparisons.

### Note on low baseline results (ResNet50)

ResNet50 is designed for ImageNet‑scale images. CIFAR‑10 images are **32×32**, so transfer learning can be unstable without careful resizing/augmentation and tuning. One hypothesis is that better performance would require **upscaling** images and revisiting augmentation + learning rate schedules (at the cost of significantly more compute on CPU).

## Notebooks

- `EDA.ipynb` — EDA + baseline transfer learning (ResNet50) + classical baselines (LogReg/CatBoost on deep features) + experiment logging.
- `NewModels.ipynb` — scratch models (ResNet18 / WideResNet) + MixUp/CutMix (tf.data) + logging.
- `PresentationAssets.ipynb` — charts/tables generated from `artifacts/experiments.csv` (no training).

## Experiment log

All runs are appended to:
- `artifacts/experiments.jsonl` (raw log)
- `artifacts/experiments.csv` (comparison table)

### Summary (best run so far)

- **exp09**: ResNet18 (scratch) + flip/crop + MixUp (α=0.2), **50k**, 50 epochs → **val_acc=0.9356**, **test_acc=0.9322**

### All experiments (from `artifacts/experiments.csv`)

`val_acc/test_acc` means the final evaluation accuracy for that run (for older runs it comes from `stage2_*` metrics).

| exp_id | exp_name | model | n_samples | split | val_acc | test_acc | logreg_test | catboost_test | aug |
|---:|---|---|---:|---|---:|---:|---:|---:|---|
| exp01 | 10k_80-10-10_noaug_baseline | ResNet50 TL | 10,000 | 80/10/10 | 0.6074 | 0.6060 | — | 0.5880 | — |
| exp02 | 10k_80-10-10_aug | ResNet50 TL | 10,000 | 80/10/10 | 0.5964 | 0.5880 | — | — | — |
| exp03 | 20k_80-10-10_noaug | ResNet50 TL | 20,000 | 80/10/10 | 0.6274 | 0.6060 | — | — | — |
| exp04 | 20k_80-10-10_aug | ResNet50 TL | 20,000 | 80/10/10 | 0.6024 | 0.5760 | — | — | — |
| exp05 | 20k_80-10-10_noaug_logreg_features | ResNet50 TL | 20,000 | 80/10/10 | 0.6557 | 0.6405 | — | — | — |
| exp06 | 20k_80-10-10_noaug_logreg_features | ResNet50 TL | 20,000 | 80/10/10 | 0.6447 | 0.6340 | 0.5500 | 0.6370 | — |
| exp07 | 50k_80-10-10_aug_head_then_ft | ResNet50 TL | 50,000 | 80/10/10 | 0.6084 | 0.5930 | 0.5540 | — | — |
| exp09 | 50k_resnet18_mixup | ResNet18 scratch | 50,000 | 80/10/10 | 0.9356 | 0.9322 | — | — | flip_crop,mixup(a=0.2) |

## Pipeline (high level)

1) Load CIFAR‑10 from a local tarball (no network) via `scripts/local_cifar10.py`  
2) Stratified split **80/10/10** inside the train set (fixed seed)  
3) Train model(s) with consistent settings and callbacks  
4) Evaluate on train/val/test splits  
5) Append metrics into `artifacts/experiments.jsonl` and export `artifacts/experiments.csv`

## One-time setup (local Jupyter)

This repository expects you to run notebooks with a dedicated Python environment so that `pip install ...` and the notebook kernel always match.

### Option A (recommended): venv + dedicated Jupyter kernel

```bash
bash scripts/setup_jupyter_env.sh
```

Then in Jupyter:
- `Kernel` → `Change kernel` → `CV Project (.venv)`

### Verify TensorFlow is visible in the notebook

Run in a notebook cell:

```python
import sys
print(sys.executable)
import tensorflow as tf
print(tf.__version__)
```

If `import tensorflow` fails, it means Jupyter is still using a different kernel/environment.

## Reproducing runs

- Baseline runs: open `EDA.ipynb`, set `EXP_ID/EXP_NAME`, run cells top-to-bottom, then run “Save experiment results”.
- New models: open `NewModels.ipynb`, choose `MODEL_TYPE` and `USE_MIXUP/USE_CUTMIX`, run top-to-bottom, then run the logging cell.
- Charts for presentation: open `PresentationAssets.ipynb` and run all (it reads `artifacts/experiments.csv`).
