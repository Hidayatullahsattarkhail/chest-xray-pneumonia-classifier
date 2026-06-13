"""
config.py — Central configuration for the Chest X-Ray Pneumonia Detection project.

This is the ONLY file you need to edit to change any setting.
All other scripts import their values from here.

Quick-start tweaks for beginners:
  - More training: increase NUM_EPOCHS (e.g. 20 for better accuracy)
  - Faster training: decrease BATCH_SIZE if you run out of memory
  - Different data folder: change DATA_DIR
  - Different checkpoint name: change CHECKPOINT
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path

# Absolute path to the chest-xray-detection/ folder.
# All other paths are derived from this so the project works regardless of
# which directory you run a command from.
_PROJECT_ROOT = Path(__file__).resolve().parent

# ── Dataset root ──────────────────────────────────────────────────────────────
# The Kaggle download extracts to a nested "chest_xray/" subfolder, so we
# auto-detect both common layouts:
#   Layout A (direct):  data/train/  data/val/  data/test/
#   Layout B (Kaggle):  data/chest_xray/train/  ...
_data_base = _PROJECT_ROOT / "data"
_kaggle_nested = _data_base / "chest_xray"
if _kaggle_nested.is_dir() and (_kaggle_nested / "train").is_dir():
    DATA_DIR = str(_kaggle_nested)   # Kaggle-style nested layout
else:
    DATA_DIR = str(_data_base)       # Standard flat layout

# Derived path — scripts that work only on the test split use this
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

# Saved model weights produced by train.py
CHECKPOINT = str(_PROJECT_ROOT / "best_model.pth")

# TorchScript export produced by export.py (used by the FastAPI server)
TORCHSCRIPT_PATH = str(_PROJECT_ROOT / "densenet121_pneumonia_prod.pt")

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

# Class labels — must match the subfolder names in data/train/, data/test/, etc.
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Input image size expected by DenseNet121 (do not change unless you retrain)
IMAGE_SIZE = 224

# ImageNet normalisation values — required when using pretrained DenseNet weights
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

# Number of full passes through the training data.
# More epochs → higher accuracy (but longer training time).
# Recommended: 10–20 for good results; 3–5 for a quick test run.
NUM_EPOCHS = 5

# Number of images processed together in one forward pass.
# Smaller batch = less GPU/CPU memory needed, but slower training.
BATCH_SIZE = 32

# Learning rate — how fast the model updates its weights.
# If accuracy is not improving, try 1e-4. If training is unstable, try 5e-4.
LR = 1e-3

# MLflow experiment name (visible in `mlflow ui`)
EXPERIMENT = "DenseNet121_Pneumonia"

# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

# Number of Grad-CAM sample images to include in the PDF report
NUM_GRADCAM = 4

# Default number of images shown in the prediction grid (visualize_results.py)
NUM_VIS_IMAGES = 12

# ─────────────────────────────────────────────────────────────────────────────
#  API SERVER  (app.py)
# ─────────────────────────────────────────────────────────────────────────────

# Host and port the FastAPI server binds to.
# Use "0.0.0.0" to accept connections from any machine (required in Docker).
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─────────────────────────────────────────────────────────────────────────────
#  API TESTING  (test_api.py)
# ─────────────────────────────────────────────────────────────────────────────

# URL of the running FastAPI server to test against
API_URL = "http://localhost:8000"

# Number of test images per class sent during API validation
NUM_TEST_IMAGES = 5

# Seconds to wait for each API response before giving up
REQUEST_TIMEOUT = 30
