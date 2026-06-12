"""
train.py — Train the chest X-ray pneumonia detection model with MLflow tracking.

Usage:
    python train.py

The script will:
  1. Load images from the data/ folder
  2. Fine-tune a pretrained DenseNet121
  3. Track all metrics in MLflow (view with: mlflow ui)
  4. Save the best model to best_model.pth
  5. Print accuracy after each epoch

View training history:
    mlflow ui
    → Open http://localhost:5000 in your browser
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from tqdm import tqdm

from model import build_model
from dataset import get_dataloaders


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR    = "data"
NUM_EPOCHS  = 5           # Increase for better accuracy (10–20 recommended)
BATCH_SIZE  = 32
LR          = 1e-3
SAVE_PATH   = "best_model.pth"
EXPERIMENT  = "DenseNet121_Pneumonia"

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Main training function ────────────────────────────────────────────────────

def train():
    # 1. Load data
    dataloaders, dataset_sizes = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )

    if "train" not in dataloaders:
        print("ERROR: No training data found in", DATA_DIR)
        return

    # 2. Build model
    model = build_model(use_pretrained=True)
    model = model.to(device)

    # 3. Loss function and optimizer
    # Only optimise the classifier head (backbone is frozen)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 4. Set up MLflow experiment
    mlflow.set_experiment(EXPERIMENT)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy      = 0.0
    start_time         = time.time()

    # 5. Training loop — all metrics logged to MLflow
    with mlflow.start_run():
        mlflow.log_param("model",       "DenseNet121")
        mlflow.log_param("epochs",      NUM_EPOCHS)
        mlflow.log_param("batch_size",  BATCH_SIZE)
        mlflow.log_param("lr",          LR)
        mlflow.log_param("device",      str(device))

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            print("-" * 30)

            for phase in ["train", "val"]:
                if phase not in dataloaders:
                    continue

                model.train() if phase == "train" else model.eval()

                running_loss    = 0.0
                running_correct = 0

                for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss    = criterion(outputs, labels)
                        preds   = outputs.argmax(dim=1)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss    += loss.item() * inputs.size(0)
                    running_correct += (preds == labels).sum().item()

                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss    / dataset_sizes[phase]
                epoch_acc  = running_correct / dataset_sizes[phase]

                print(f"{phase:5s}  Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.4f}")

                # Log to MLflow
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                mlflow.log_metric(f"{phase}_acc",  epoch_acc,  step=epoch)

                # Save best model based on validation accuracy
                if phase == "val" and epoch_acc > best_accuracy:
                    best_accuracy      = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())
                    torch.save(best_model_weights, SAVE_PATH)
                    print(f"  ✓ New best model saved (val acc: {best_accuracy:.4f})")

        mlflow.log_param("best_val_acc", f"{best_accuracy:.4f}")
        mlflow.log_artifact(SAVE_PATH)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    print(f"\nView run history: mlflow ui")


if __name__ == "__main__":
    train()
