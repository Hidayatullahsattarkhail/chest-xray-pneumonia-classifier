"""
evaluate.py — Evaluate the trained model on the test set.

Usage:
    python evaluate.py

What it prints and saves:
  - Overall accuracy, precision, recall, F1-score
  - Per-class accuracy (NORMAL and PNEUMONIA separately)
  - Confusion matrix
  - ROC curve saved as roc_curve.png

Make sure you have already run train.py so that best_model.pth exists.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from model import load_model
from dataset import get_dataloaders, CLASS_NAMES


# ── Configuration ─────────────────────────────────────────────────────────────

from config import DATA_DIR, CHECKPOINT, BATCH_SIZE

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Helper: print a readable confusion matrix ─────────────────────────────────

def print_confusion_matrix(cm, class_names):
    col_width = 12
    print("\n── Confusion Matrix ─────────────────────────────────")
    print(f"{'':>{col_width}}", end="")
    for name in class_names:
        print(f"{'Pred: ' + name:>{col_width}}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{'True: ' + name:>{col_width}}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>{col_width}}", end="")
        print()
    print()
    print("  Diagonal values = correct predictions ✓")
    print("  Off-diagonal    = mistakes            ✗")


# ── Helper: plot and save ROC curve ──────────────────────────────────────────

def plot_roc_curve(all_labels, all_probs, save_path="roc_curve.png"):
    """
    Plot the ROC (Receiver Operating Characteristic) curve.

    The ROC curve shows the trade-off between:
      True Positive Rate  (how many pneumonia cases we correctly catch)
      False Positive Rate (how many normal cases we wrongly flag)

    AUC (Area Under Curve) closer to 1.0 = better model.
    AUC = 0.5 means the model is no better than random guessing.
    """
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
            label="Random classifier (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Pneumonia Detection", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve saved to: {save_path}")
    return roc_auc


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
    all_probs  = []   # Probability of PNEUMONIA (class index 1) for ROC curve

    print("\nRunning predictions on test set...")
    for inputs, labels in tqdm(dataloaders["test"]):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # PNEUMONIA probability

    # 4. Compute metrics
    overall_acc = accuracy_score(all_labels, all_preds)
    precision   = precision_score(all_labels, all_preds)
    recall      = recall_score(all_labels, all_preds)
    f1          = f1_score(all_labels, all_preds)
    cm          = confusion_matrix(all_labels, all_preds)

    # 5. Print results
    print("\n" + "=" * 52)
    print("  EVALUATION RESULTS")
    print("=" * 52)
    print(f"\n  Total test images : {dataset_sizes['test']}")
    print(f"  Overall Accuracy  : {overall_acc * 100:.2f}%")
    print(f"  Precision         : {precision:.3f}")
    print(f"  Recall            : {recall:.3f}")
    print(f"  F1-Score          : {f1:.3f}")

    print("\n── Per-Class Accuracy ───────────────────────────────")
    for i, name in enumerate(CLASS_NAMES):
        total_in_class   = cm[i].sum()
        correct_in_class = cm[i][i]
        class_acc        = correct_in_class / total_in_class if total_in_class > 0 else 0
        print(f"  {name:<12}  {correct_in_class:>4} / {total_in_class:>4}  →  {class_acc * 100:.2f}%")

    print_confusion_matrix(cm, CLASS_NAMES)

    print("── Full Report ──────────────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3))

    print("Terms explained:")
    print("  Precision — of all images predicted as X, how many were actually X?")
    print("  Recall    — of all actual X images, how many did the model find?")
    print("  F1-score  — balance between precision and recall (higher is better)")

    # 6. ROC curve
    print("\n── ROC Curve ────────────────────────────────────────")
    roc_auc = plot_roc_curve(all_labels, all_probs)
    print(f"  AUC Score: {roc_auc:.3f}  (1.0 = perfect, 0.5 = random)")

    print("=" * 52 + "\n")


if __name__ == "__main__":
    evaluate()
