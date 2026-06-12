"""
train.py — Train the chest X-ray pneumonia detection model.

Usage:
    python train.py

The script will:
  1. Load images from the data/ folder
  2. Fine-tune a pretrained ResNet18
  3. Save the best model to best_model.pth
  4. Print accuracy after each epoch
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import build_model
from dataset import get_dataloaders


# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR    = "data"       # Folder with train/val/test subfolders
NUM_EPOCHS  = 10           # Training epochs (increase for better accuracy)
BATCH_SIZE  = 32
LR          = 1e-3         # Learning rate
SAVE_PATH   = "best_model.pth"

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
    # Only optimise the final layers (the rest are frozen in model.py)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    # Reduce LR if validation loss stops improving
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 4. Training loop
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase not in dataloaders:
                continue

            model.train() if phase == "train" else model.eval()

            running_loss    = 0.0
            running_correct = 0

            # Iterate over batches
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)

                    # Backward pass only during training
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

            # Save best model based on validation accuracy
            if phase == "val" and epoch_acc > best_accuracy:
                best_accuracy       = epoch_acc
                best_model_weights  = copy.deepcopy(model.state_dict())
                torch.save(best_model_weights, SAVE_PATH)
                print(f"  ✓ New best model saved (val acc: {best_accuracy:.4f})")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()
