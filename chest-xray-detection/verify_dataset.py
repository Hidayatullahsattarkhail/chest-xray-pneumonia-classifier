"""
verify_dataset.py — Dataset structure checker for Chest X-Ray Pneumonia project.

Checks that the dataset is in place and correctly structured for PyTorch ImageFolder.
Run this any time after uploading or moving the dataset to confirm everything is wired up.

Usage:
    xray-verify              (after pip install -e .)
    python verify_dataset.py (from the chest-xray-detection/ folder)
"""

import os
import sys
from pathlib import Path


# ── ANSI colour helpers ───────────────────────────────────────────────────────

USE_COLOUR = sys.platform != "win32"
GREEN  = "\033[92m" if USE_COLOUR else ""
YELLOW = "\033[93m" if USE_COLOUR else ""
RED    = "\033[91m" if USE_COLOUR else ""
BOLD   = "\033[1m"  if USE_COLOUR else ""
RESET  = "\033[0m"  if USE_COLOUR else ""

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"     {msg}")


# ── Verification logic ────────────────────────────────────────────────────────

def verify() -> bool:
    """
    Run all dataset checks. Returns True if everything is ready, False otherwise.
    """
    print(f"\n{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    print(f"{BOLD}  Chest X-Ray Dataset Integration Checker{RESET}")
    print(f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}\n")

    all_ok = True

    # ── 1. Load config ────────────────────────────────────────────────────────
    print(f"{BOLD}[1] Configuration{RESET}")
    try:
        from config import (
            DATA_DIR, TEST_DATA_DIR, CHECKPOINT, TORCHSCRIPT_PATH,
            CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS,
            _PROJECT_ROOT,
        )
        ok(f"config.py loaded — PROJECT_ROOT: {_PROJECT_ROOT}")
        info(f"DATA_DIR          = {DATA_DIR}")
        info(f"TEST_DATA_DIR     = {TEST_DATA_DIR}")
        info(f"CHECKPOINT        = {CHECKPOINT}")
        info(f"TORCHSCRIPT_PATH  = {TORCHSCRIPT_PATH}")
        info(f"CLASS_NAMES       = {CLASS_NAMES}")
        info(f"IMAGE_SIZE        = {IMAGE_SIZE}  BATCH_SIZE = {BATCH_SIZE}  NUM_EPOCHS = {NUM_EPOCHS}")
    except Exception as exc:
        fail(f"config.py failed to load: {exc}")
        return False
    print()

    # ── 2. Dataset root ───────────────────────────────────────────────────────
    print(f"{BOLD}[2] Dataset Root{RESET}")
    data_path = Path(DATA_DIR)

    # Check for common wrong locations
    project_root = Path(CHECKPOINT).parent
    kaggle_nested = project_root / "data" / "chest_xray"
    bare_data     = project_root / "data"

    if data_path.is_dir():
        ok(f"DATA_DIR exists: {data_path}")
        if "chest_xray" in str(data_path):
            ok("Kaggle-style nested layout detected (data/chest_xray/) — auto-handled")
    else:
        fail(f"DATA_DIR not found: {data_path}")
        all_ok = False

        # Offer hints
        if bare_data.is_dir():
            warn(f"Found {bare_data}/ but train/val/test subfolders may be missing inside it")
        else:
            print()
            print(f"  {BOLD}How to fix:{RESET}")
            print(f"  Upload the dataset so the folder structure looks like this:\n")
            print(f"    chest-xray-detection/")
            print(f"    └── data/")
            print(f"        ├── train/")
            print(f"        │   ├── NORMAL/      (contains .jpeg images)")
            print(f"        │   └── PNEUMONIA/   (contains .jpeg images)")
            print(f"        ├── val/")
            print(f"        │   ├── NORMAL/")
            print(f"        │   └── PNEUMONIA/")
            print(f"        └── test/")
            print(f"            ├── NORMAL/")
            print(f"            └── PNEUMONIA/")
            print()
            print(f"  If the Kaggle zip creates a 'chest_xray/' subfolder, that is fine —")
            print(f"  config.py will auto-detect it as long as it is inside 'data/'.")
            print()
    print()

    # ── 3. Splits ─────────────────────────────────────────────────────────────
    print(f"{BOLD}[3] Train / Val / Test Splits{RESET}")
    splits = {"train": 0, "val": 0, "test": 0}
    splits_ok = True

    if data_path.is_dir():
        for split in splits:
            split_path = data_path / split
            if split_path.is_dir():
                count = sum(1 for _ in split_path.rglob("*") if _.is_file())
                splits[split] = count
                ok(f"{split:5s}/ found — {count:,} files total")
            else:
                fail(f"{split:5s}/ not found inside {data_path}")
                splits_ok = False
                all_ok = False
    else:
        for split in splits:
            fail(f"{split}/ — skipped (DATA_DIR missing)")
        splits_ok = False
        all_ok = False
    print()

    # ── 4. Class folders ──────────────────────────────────────────────────────
    print(f"{BOLD}[4] Class Folders (ImageFolder compatibility){RESET}")
    expected_classes = CLASS_NAMES  # ["NORMAL", "PNEUMONIA"]

    if data_path.is_dir() and splits_ok:
        for split in ["train", "val", "test"]:
            split_path = data_path / split
            if not split_path.is_dir():
                continue
            found_classes = sorted(
                d.name for d in split_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
            missing = [c for c in expected_classes if c not in found_classes]
            extra   = [c for c in found_classes if c not in expected_classes]

            if not missing and not extra:
                counts = {
                    c: len(list((split_path / c).glob("*")))
                    for c in expected_classes
                }
                ok(f"{split:5s}/ classes OK — {counts}")
            else:
                if missing:
                    fail(f"{split:5s}/ — missing class folders: {missing}")
                    all_ok = False
                if extra:
                    warn(f"{split:5s}/ — unexpected folders (ignored by ImageFolder): {extra}")
    else:
        warn("Skipping class check — dataset root not found")
    print()

    # ── 5. Sample image readability ───────────────────────────────────────────
    print(f"{BOLD}[5] Image Readability{RESET}")
    if data_path.is_dir():
        test_path = data_path / "test"
        sample = None
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            found = list(test_path.rglob(ext))
            if found:
                sample = found[0]
                break

        if sample:
            try:
                from PIL import Image
                img = Image.open(sample).convert("RGB")
                ok(f"Sample image readable: {sample.name}  size={img.size}  mode={img.mode}")
            except Exception as exc:
                fail(f"Could not open sample image {sample}: {exc}")
                all_ok = False
        else:
            warn("No .jpeg/.jpg/.png images found in test/ — is the dataset empty?")
            all_ok = False
    else:
        warn("Skipping image check — dataset root not found")
    print()

    # ── 6. DataLoader smoke-test ──────────────────────────────────────────────
    print(f"{BOLD}[6] DataLoader Smoke Test{RESET}")
    if data_path.is_dir() and splits_ok:
        try:
            from dataset import get_dataloaders
            loaders, sizes = get_dataloaders(DATA_DIR, batch_size=4, num_workers=0)
            for split, loader in loaders.items():
                batch = next(iter(loader))
                imgs, labels = batch
                ok(f"{split:5s} — batch shape: {list(imgs.shape)}  labels: {labels.tolist()[:4]}")
        except Exception as exc:
            fail(f"DataLoader failed: {exc}")
            all_ok = False
    else:
        warn("Skipping DataLoader test — dataset not present")
    print()

    # ── 7. Model checkpoint ───────────────────────────────────────────────────
    print(f"{BOLD}[7] Model Checkpoint{RESET}")
    ckpt_path = Path(CHECKPOINT)
    if ckpt_path.is_file():
        size_mb = ckpt_path.stat().st_size / (1024 ** 2)
        ok(f"best_model.pth found — {size_mb:.1f} MB")
    else:
        warn(f"best_model.pth not found at: {CHECKPOINT}")
        info("Run 'xray-train' to create it.")
    print()

    # ── 8. TorchScript export ─────────────────────────────────────────────────
    print(f"{BOLD}[8] TorchScript Export{RESET}")
    ts_path = Path(TORCHSCRIPT_PATH)
    if ts_path.is_file():
        size_mb = ts_path.stat().st_size / (1024 ** 2)
        ok(f"TorchScript model found — {size_mb:.1f} MB")
    else:
        warn(f"TorchScript model not found at: {TORCHSCRIPT_PATH}")
        info("Run 'xray-export' to create it (optional — FastAPI falls back to .pth).")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    if all_ok:
        print(f"{GREEN}{BOLD}  ✓ Dataset integration successful! All checks passed.{RESET}")
        print(f"\n  You can now run:")
        print(f"    xray-evaluate   — test set metrics")
        print(f"    xray-predict    — single image prediction")
        print(f"    xray-visualize  — prediction grid")
        print(f"    xray-gradcam    — Grad-CAM heatmap")
        print(f"    xray-report     — generate PDF report")
        print(f"    xray-demo       — interactive menu")
    else:
        print(f"{RED}{BOLD}  ✗ Dataset integration incomplete.{RESET}")
        print(f"\n  Fix the issues above, then run 'xray-verify' again.")
    print(f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}\n")

    return all_ok


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
