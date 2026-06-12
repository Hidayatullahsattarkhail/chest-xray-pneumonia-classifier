# Chest X-Ray Pneumonia Detection

A simple PyTorch image classifier that detects **NORMAL** vs **PNEUMONIA** in chest X-ray images using a pretrained ResNet18.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Get your Kaggle API key from https://www.kaggle.com/settings → API → Create New Token, place `kaggle.json` at `~/.kaggle/kaggle.json`, then run:

```bash
python download_dataset.py
```

This downloads and extracts the [chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset into a `data/` folder.

Expected folder structure:
```
data/
├── train/
│   ├── NORMAL/        (1341 images)
│   └── PNEUMONIA/     (3875 images)
├── val/
│   ├── NORMAL/        (8 images)
│   └── PNEUMONIA/     (8 images)
└── test/
    ├── NORMAL/        (234 images)
    └── PNEUMONIA/     (390 images)
```

### 3. Train the model

```bash
python train.py
```

- Trains for 10 epochs by default
- Saves the best model to `best_model.pth`
- Prints loss and accuracy after each epoch

### 4. Predict on a new image

```bash
python predict.py data/test/NORMAL/IM-0001-0001.jpeg
```

Output:
```
========================================
  Prediction : NORMAL
  Confidence : 98.3%
========================================
```

Or use it in your own code:

```python
from predict import predict

label, confidence = predict("my_xray.jpg")
print(f"{label} — {confidence * 100:.1f}% confident")
```

---

## Files

| File | What it does |
|---|---|
| `model.py` | Defines the ResNet18 classifier |
| `dataset.py` | Loads images and applies transforms |
| `train.py` | Training loop with validation |
| `predict.py` | Single-image inference function + CLI |
| `download_dataset.py` | Downloads the Kaggle dataset |

---

## How it works

1. **Model**: Pretrained ResNet18 (ImageNet weights). All layers are frozen except a new final classifier (`512 → 128 → 2`).
2. **Training**: Fine-tuning with Adam optimizer and CrossEntropy loss. Augmentation (flip, rotation, brightness) is applied to the training set only.
3. **Classes**: Index `0` = NORMAL, Index `1` = PNEUMONIA.
4. **GPU**: Automatically uses CUDA if available, otherwise CPU.

---

## Tips

- The Kaggle dataset has very few validation images (8 per class). Val accuracy may be noisy — watch training loss as well.
- Increase `NUM_EPOCHS` in `train.py` for better results (20–30 is common).
- Unfreeze more ResNet layers for deeper fine-tuning once the head is trained.
