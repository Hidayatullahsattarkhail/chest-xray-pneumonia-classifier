"""
download_dataset.py — Download the Kaggle chest-xray-pneumonia dataset.

Before running, set up your Kaggle credentials:
  1. Go to https://www.kaggle.com/settings → API → Create New Token
  2. Download kaggle.json
  3. Place it at ~/.kaggle/kaggle.json   (Linux/Mac)
               or  C:/Users/<you>/.kaggle/kaggle.json  (Windows)
  4. Run: python download_dataset.py

The dataset will be extracted to: data/
"""

import os
import zipfile

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
DOWNLOAD_DIR   = "."
DATA_DIR       = "data"


def download():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("  Run: pip install kaggle")
        return

    print("Downloading dataset from Kaggle...")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} -p {DOWNLOAD_DIR} --unzip"
    )

    # The dataset unzips to chest_xray/ — rename to data/ for our scripts
    extracted = os.path.join(DOWNLOAD_DIR, "chest_xray")
    if os.path.isdir(extracted) and not os.path.isdir(DATA_DIR):
        os.rename(extracted, DATA_DIR)
        print(f"Dataset extracted to: {DATA_DIR}/")
    elif os.path.isdir(DATA_DIR):
        print(f"Dataset already exists at: {DATA_DIR}/")
    else:
        print("Could not find extracted folder. Check download output above.")
        return

    # Verify structure
    for split in ["train", "val", "test"]:
        path = os.path.join(DATA_DIR, split)
        if os.path.isdir(path):
            classes = os.listdir(path)
            counts  = {c: len(os.listdir(os.path.join(path, c))) for c in classes}
            print(f"  {split}: {counts}")
        else:
            print(f"  Warning: '{split}' folder not found in {DATA_DIR}/")

    print("\nDataset ready! You can now run: python train.py")


if __name__ == "__main__":
    download()
