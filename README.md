# CIFAR-10 Image Classification: Baseline → CIFAR-friendly models + MixUp/CutMix

This is a small CV project focused on building a **clean, reproducible experiment pipeline** for image classification on **CIFAR-10** and comparing modeling choices.

**Main notebooks**
- `EDA.ipynb` — dataset EDA + baseline transfer learning (ResNet50) + classical baselines (LogReg/CatBoost on deep features) + experiment logging
- `NewModels.ipynb` — CIFAR-friendly scratch models (**ResNet18**, **WideResNet**) + **MixUp/CutMix** (tf.data) + logging
- `PresentationAssets.ipynb` — slide-ready charts/tables generated from `artifacts/experiments.csv` (no training)

## Results (quick summary)

All runs are appended to:
- `artifacts/experiments.jsonl` (raw log)
- `artifacts/experiments.csv` (table for comparison)

| Exp | Model | Data | Augmentations | Val acc | Test acc | Notes |
|---:|---|---:|---|---:|---:|---|
| exp06 | ResNet50 (transfer learning) | 20,000 | — | 0.6447 | 0.6340 | Best TL baseline (2-stage) |
| exp07 | ResNet50 (transfer learning) | 50,000 | — | 0.6084 | 0.5930 | Scale-up attempt (worse) |
| exp09 | ResNet18 (scratch) | 50,000 | flip/crop + MixUp (α=0.2) | 0.9356 | 0.9322 | Final / best run so far |

## Pipeline (high-level)

1) Load CIFAR-10 from a local tarball (no network) via `scripts/local_cifar10.py`  
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
