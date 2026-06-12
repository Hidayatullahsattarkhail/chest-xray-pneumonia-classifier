"""
evaluate.py — Evaluate the trained model on the test dataset.

Usage:
    python evaluate.py

What it prints:
  - Overall accuracy
  - Per-class accuracy (NORMAL and PNEUMONIA separately)
  - Confusion matrix showing correct vs incorrect predictions

Make sure you have already run train.py so that best_model.pth exists.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from model import load_model
from dataset import get_dataloaders, CLASS_NAMES


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = "data"
CHECKPOINT = "best_model.pth"
BATCH_SIZE = 32

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Helper: print a readable confusion matrix ─────────────────────────────────

def print_confusion_matrix(cm, class_names):
    """
    Print a confusion matrix in a plain, easy-to-read format.

    Rows   = Actual labels
    Columns = Predicted labels
    """
    col_width = 12

    print("\n── Confusion Matrix ─────────────────────────────────")
    print(f"{'':>{col_width}}", end="")
    for name in class_names:
        print(f"{'Pred: ' + name:>{col_width}}", end="")
    print()

    for i, name in enumerate(class_names):
        print(f"{'True: ' + name:>{col_width}}", end="")
        for j in range(len(class_names)):
            cell = cm[i][j]
            print(f"{cell:>{col_width}}", end="")
        print()

    print()
    print("How to read this:")
    print(f"  Row = what the image actually is  (True label)")
    print(f"  Col = what the model predicted    (Predicted label)")
    print(f"  Diagonal values = correct predictions ✓")
    print(f"  Off-diagonal    = mistakes        ✗")


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate():
    # 1. Load test data
    dataloaders, dataset_sizes = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )

    if "test" not in dataloaders:
        print(f"ERROR: No test data found in '{DATA_DIR}/test/'")
        return

    # 2. Load trained model
    model = load_model(CHECKPOINT, device)

    # 3. Run predictions on the entire test set
    all_preds  = []
    all_labels = []

    print("\nRunning predictions on test set...")
    for inputs, labels in tqdm(dataloaders["test"]):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    # 4. Overall accuracy
    overall_acc = accuracy_score(all_labels, all_preds)

    print("\n" + "=" * 52)
    print("  EVALUATION RESULTS")
    print("=" * 52)
    print(f"\n  Overall Accuracy: {overall_acc * 100:.2f}%")
    print(f"  Total test images: {dataset_sizes['test']}")

    # 5. Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)

    print("\n── Per-Class Accuracy ───────────────────────────────")
    for i, name in enumerate(CLASS_NAMES):
        total_in_class   = cm[i].sum()
        correct_in_class = cm[i][i]
        class_acc        = correct_in_class / total_in_class if total_in_class > 0 else 0
        print(f"  {name:<12}  {correct_in_class:>4} / {total_in_class:>4}  →  {class_acc * 100:.2f}%")

    # 6. Confusion matrix
    print_confusion_matrix(cm, CLASS_NAMES)

    # 7. Full scikit-learn report (precision, recall, F1)
    print("── Detailed Report ──────────────────────────────────")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=3,
    )
    print(report)

    print("Terms explained:")
    print("  Precision — of all images predicted as X, how many were actually X?")
    print("  Recall    — of all actual X images, how many did the model find?")
    print("  F1-score  — balance between precision and recall (higher is better)")
    print("  Support   — number of test images in each class")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    evaluate()
