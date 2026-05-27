# CV project (CIFAR-10 + ResNet50 transfer learning)

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
