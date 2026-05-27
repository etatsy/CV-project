#!/usr/bin/env bash
set -euo pipefail

# Creates a local virtual environment, installs dependencies, and registers a Jupyter kernel.
# Run from repo root:  bash scripts/setup_jupyter_env.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
KERNEL_NAME="${KERNEL_NAME:-cv-project}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-CV Project (.venv)}"

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -V

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Upgrading pip/setuptools/wheel ..."
"$VENV_DIR/bin/python" -m pip install -U pip setuptools wheel

echo "Installing project requirements ..."
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo "Installing Jupyter + ipykernel ..."
"$VENV_DIR/bin/python" -m pip install -U jupyter ipykernel

echo "Registering Jupyter kernel: $KERNEL_NAME"
"$VENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo
echo "Done."
echo "Next:"
echo "1) Restart Jupyter."
echo "2) In the notebook: Kernel -> Change kernel -> \"$KERNEL_DISPLAY_NAME\""
echo "3) Re-run the imports cell."
