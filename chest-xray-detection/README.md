# Chest X-Ray Pneumonia Detection

A beginner-friendly PyTorch project that detects **NORMAL vs PNEUMONIA** in chest X-rays using **DenseNet121** — complete with training, evaluation, Grad-CAM explainability, a FastAPI server, and a PDF report generator.

---

## Quick Start (two ways)

### Option A — Interactive demo menu

```bash
cd chest-xray-detection
pip install -e .      # one-time install (registers all xray-* commands)
xray-demo             # launches the interactive menu
```

The demo walks you through every step with numbered options.

### Option B — Install then use individual commands

```bash
cd chest-xray-detection
pip install -e .      # one-time setup
```

Then use the commands below from any directory.

---

## All Commands

| Command | What it does |
|---|---|
| `xray-download` | Download the Kaggle dataset into `data/` |
| `xray-train` | Fine-tune DenseNet121, saves `best_model.pth` |
| `xray-evaluate` | Accuracy, F1, confusion matrix, ROC curve |
| `xray-predict <image>` | Predict one X-ray (NORMAL / PNEUMONIA + confidence) |
| `xray-visualize` | Grid of test predictions with colour-coded correctness |
| `xray-gradcam <image>` | Grad-CAM heatmap for one image (shows _why_ the model decided) |
| `xray-compare` | Side-by-side Grad-CAM comparison across both classes |
| `xray-report` | Generate a full PDF report (`xray_report.pdf`) |
| `xray-export` | Export model to TorchScript for production |
| `xray-serve` | Start the FastAPI prediction server |
| `xray-test-api` | Validate the running API against test images |
| `xray-demo` | Interactive menu tying all of the above together |

**Examples:**

```bash
xray-predict data/test/PNEUMONIA/person1_virus_1.jpeg
xray-gradcam data/test/NORMAL/IM-0001-0001.jpeg
xray-visualize --num 16 --only-wrong
xray-compare --num-per-class 3
xray-serve --port 9000
xray-test-api --num 20
```

---

## Step-by-Step Walkthrough

### 1. Install dependencies

```bash
cd chest-xray-detection
pip install -e .
```

### 2. Download the dataset

Get your Kaggle API key from https://www.kaggle.com/settings → **API → Create New Token**, place `kaggle.json` at `~/.kaggle/kaggle.json`, then:

```bash
xray-download
```

Expected folder structure after download:
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
xray-train
```

- Fine-tunes DenseNet121 (pretrained on ImageNet)
- Saves the best weights to `best_model.pth`
- Logs metrics to MLflow (run `mlflow ui` to see them)
- Default: 5 epochs — increase `NUM_EPOCHS` in `config.py` for better accuracy

### 4. Evaluate

```bash
xray-evaluate
```

Prints accuracy, precision, recall, F1 — and opens a confusion matrix + ROC curve.

### 5. Predict a single image

```bash
xray-predict data/test/NORMAL/IM-0001-0001.jpeg
```

Output:
```
========================================
  Prediction : NORMAL
  Confidence : 98.3%
========================================
```

### 6. Visualise predictions

```bash
xray-visualize            # 12 random test images
xray-visualize --num 24   # 24 images
xray-visualize --only-wrong  # only mistakes
```

### 7. Grad-CAM explainability

```bash
xray-gradcam data/test/PNEUMONIA/person1_virus_1.jpeg
xray-compare                  # auto-picks 2 images per class
xray-compare --num-per-class 4
```

Grad-CAM highlights the pixels that most influenced the prediction — useful for sanity-checking that the model is looking at the lungs.

### 8. Generate a PDF report

```bash
xray-report
```

Creates `xray_report.pdf` with metrics, confusion matrix, ROC curve, and Grad-CAM samples.

### 9. Export & deploy

```bash
xray-export        # save as TorchScript (.pt)
xray-serve         # start FastAPI on http://localhost:8000
```

API docs available at http://localhost:8000/docs once the server is running.

```bash
xray-test-api      # validate predictions against test images
```

---

## Configuration

**All settings live in `config.py`** — it is the only file you ever need to edit.

```python
# Key settings (config.py)
NUM_EPOCHS = 5      # ↑ increase for better accuracy (10–20 recommended)
BATCH_SIZE = 32     # ↓ decrease if you run out of memory
LR         = 1e-3   # learning rate

DATA_DIR   = "data"          # root dataset folder
CHECKPOINT = "best_model.pth"
```

---

## Files

| File | Purpose |
|---|---|
| `config.py` | **Central settings** — edit this to change anything |
| `cli.py` | Entry points for the `xray-*` shell commands |
| `model.py` | DenseNet121 classifier + Grad-CAM target layer |
| `dataset.py` | Image loading, augmentation, dataloaders |
| `train.py` | Training loop with MLflow tracking |
| `evaluate.py` | Accuracy, F1, confusion matrix, ROC curve |
| `predict.py` | Single-image inference |
| `gradcam.py` | Grad-CAM heatmap generation |
| `compare_gradcam.py` | Side-by-side Grad-CAM comparison |
| `visualize_results.py` | Prediction grid with correctness colour coding |
| `report.py` | PDF report generator |
| `export.py` | TorchScript export |
| `app.py` | FastAPI prediction server |
| `test_api.py` | API validation script |
| `download_dataset.py` | Kaggle dataset downloader |
| `demo.py` | Interactive menu |
| `pyproject.toml` | Package definition (enables `xray-*` commands) |
| `Dockerfile` | Container for cloud deployment |
| `requirements.txt` | Dependency list |

---

## How It Works

1. **Model**: DenseNet121 pretrained on ImageNet. The classifier head is replaced with a 2-class output layer and the whole network is fine-tuned.
2. **Training**: Adam optimiser, CrossEntropy loss. Training images are augmented (random flip, rotation, colour jitter). The best validation-accuracy checkpoint is saved.
3. **Explainability**: Grad-CAM uses gradients flowing into the last DenseNet dense block to produce a heatmap showing which regions drove the prediction.
4. **Deployment**: The model is exported to TorchScript (no Python class needed to load it) and served through FastAPI with a `/predict` endpoint that accepts an uploaded image.

---

## Docker Deployment

```bash
docker build -t xray-pneumonia .
docker run -p 8000:8000 xray-pneumonia
```

Then open http://localhost:8000/docs for the interactive API.

---

## Tips

- The val split is very small (8 images per class). Val accuracy can be noisy — watch training loss too.
- For best accuracy: set `NUM_EPOCHS = 20` in `config.py` before training.
- Grad-CAM should highlight lung regions — if it's highlighting the edges of the image, the model may need more training.
- Run `mlflow ui` after training to compare experiments visually.
