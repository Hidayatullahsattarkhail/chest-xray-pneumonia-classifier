"""
report.py — Generate a PDF summary report of the model's performance.

Usage:
    python report.py

Output:
    xray_report.pdf — a structured report containing:
      - Project overview
      - Accuracy, precision, recall, F1
      - Confusion matrix
      - ROC curve
      - Sample Grad-CAM heatmaps (2–4 images)
      - Plain-English performance summary

Make sure best_model.pth exists (run train.py first).
"""

import os
import io
import random
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm as rl_cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle, HRFlowable,
)

from model import load_model
from dataset import get_dataloaders, get_transforms, CLASS_NAMES
from gradcam import generate_heatmap


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR    = "data"
CHECKPOINT  = "best_model.pth"
OUTPUT_PDF  = "xray_report.pdf"
BATCH_SIZE  = 32
NUM_GRADCAM = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Colours
BLUE       = colors.HexColor("#2c3e50")
LIGHT_BLUE = colors.HexColor("#3498db")
GREEN      = colors.HexColor("#27ae60")
LIGHT_GREY = colors.HexColor("#ecf0f1")
MID_GREY   = colors.HexColor("#bdc3c7")
WHITE      = colors.white


# ── Step 1: Collect evaluation metrics ───────────────────────────────────────

def collect_metrics() -> dict:
    print("  Running evaluation on test set...")
    dataloaders, dataset_sizes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    model = load_model(CHECKPOINT, device)

    all_preds, all_labels, all_probs = [], [], []
    for inputs, labels in tqdm(dataloaders["test"], desc="  Evaluating"):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())

    cm_array   = confusion_matrix(all_labels, all_preds)
    report_str = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)

    class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        total = cm_array[i].sum()
        class_acc[name] = cm_array[i][i] / total if total > 0 else 0.0

    return {
        "overall_acc":   accuracy_score(all_labels, all_preds),
        "precision":     precision_score(all_labels, all_preds),
        "recall":        recall_score(all_labels, all_preds),
        "f1":            f1_score(all_labels, all_preds),
        "class_acc":     class_acc,
        "cm":            cm_array,
        "report_str":    report_str,
        "dataset_sizes": dataset_sizes,
        "fpr":           fpr,
        "tpr":           tpr,
        "roc_auc":       roc_auc,
    }


# ── Step 2: Confusion matrix PNG ─────────────────────────────────────────────

def render_confusion_matrix(cm_array: np.ndarray) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_array, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    total = cm_array.sum()
    for i in range(2):
        for j in range(2):
            count   = cm_array[i][j]
            percent = count / total * 100
            color   = "white" if count > cm_array.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({percent:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    color=color, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf


# ── Step 3: ROC curve PNG ─────────────────────────────────────────────────────

def render_roc_curve(fpr, tpr, roc_auc: float) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
            label="Random classifier (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Pneumonia Detection", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf


# ── Step 4: Grad-CAM grid PNG ─────────────────────────────────────────────────

def render_gradcam_grid(num_images: int = NUM_GRADCAM) -> io.BytesIO:
    print(f"  Generating {num_images} Grad-CAM samples...")
    model = load_model(CHECKPOINT, device)

    samples = []
    per_class = max(1, num_images // len(CLASS_NAMES))
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, "test", class_name)
        if not os.path.isdir(class_dir):
            continue
        files  = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        chosen = random.sample(files, min(per_class, len(files)))
        for fname in chosen:
            samples.append((os.path.join(class_dir, fname), class_name))
    samples = samples[:num_images]

    rows = len(samples)
    fig, axes = plt.subplots(rows, 2, figsize=(9, rows * 3.6))
    if rows == 1:
        axes = [axes]

    fig.suptitle("Grad-CAM Heatmaps — Where the Model Focused",
                 fontsize=13, fontweight="bold", y=1.01)

    for row_idx, (path, true_label) in enumerate(samples):
        r = generate_heatmap(model, path, device)
        is_correct   = (r["pred_label"] == true_label)
        label_color  = "#27ae60" if is_correct else "#e74c3c"
        status       = "✓" if is_correct else "✗"

        axes[row_idx][0].imshow(r["original"], cmap="gray")
        axes[row_idx][0].set_title(f"True: {true_label}", fontsize=9, color="#555")
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(r["overlaid"])
        axes[row_idx][1].set_title(
            f"{status} Pred: {r['pred_label']}  ({r['confidence']*100:.1f}%)",
            fontsize=9, color=label_color, fontweight="bold")
        axes[row_idx][1].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf


# ── Step 5: Build summary text ────────────────────────────────────────────────

def build_summary(metrics: dict) -> str:
    acc    = metrics["overall_acc"] * 100
    n_acc  = metrics["class_acc"]["NORMAL"]    * 100
    p_acc  = metrics["class_acc"]["PNEUMONIA"] * 100
    total  = metrics["dataset_sizes"].get("test", "?")
    auc_sc = metrics["roc_auc"]
    rating = "excellent" if acc >= 90 else "good" if acc >= 80 else "moderate"
    return (
        f"The model achieved {rating} overall accuracy of {acc:.1f}% on {total} unseen test images, "
        f"with an AUC score of {auc_sc:.3f} (1.0 = perfect). "
        f"For NORMAL cases it correctly identified {n_acc:.1f}% of images, "
        f"and for PNEUMONIA cases {p_acc:.1f}%. "
        f"Precision of {metrics['precision']:.3f} and recall of {metrics['recall']:.3f} "
        f"reflect the trade-off between false positives and false negatives. "
        f"The Grad-CAM heatmaps show the model focusing on lung field regions — "
        f"red/orange areas indicate the most influential parts of each X-ray."
    )


# ── Step 6: Assemble the PDF ──────────────────────────────────────────────────

def generate_report():
    print("\n" + "=" * 52)
    print("  Generating PDF Report")
    print("=" * 52)

    if not os.path.isfile(CHECKPOINT):
        print(f"\nERROR: '{CHECKPOINT}' not found. Run train.py first.")
        return
    if not os.path.isdir(os.path.join(DATA_DIR, "test")):
        print(f"\nERROR: Test data not found. Run download_dataset.py first.")
        return

    print("\n[1/4] Collecting evaluation metrics...")
    metrics = collect_metrics()

    print("\n[2/4] Rendering confusion matrix and ROC curve...")
    cm_buf  = render_confusion_matrix(metrics["cm"])
    roc_buf = render_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"])

    print("\n[3/4] Generating Grad-CAM heatmap samples...")
    gc_buf = render_gradcam_grid(num_images=NUM_GRADCAM)

    print("\n[4/4] Building PDF...")
    summary_text = build_summary(metrics)

    # ── PDF styles ────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle("T", parent=styles["Title"], fontSize=22,
                                 textColor=BLUE, spaceAfter=6, alignment=TA_CENTER)
    style_sub   = ParagraphStyle("S", parent=styles["Normal"], fontSize=11,
                                 textColor=colors.HexColor("#7f8c8d"), spaceAfter=4,
                                 alignment=TA_CENTER)
    style_sec   = ParagraphStyle("H", parent=styles["Heading1"], fontSize=13,
                                 textColor=BLUE, spaceBefore=16, spaceAfter=6)
    style_body  = ParagraphStyle("B", parent=styles["Normal"], fontSize=10,
                                 leading=15, textColor=colors.HexColor("#2c3e50"),
                                 alignment=TA_JUSTIFY, spaceAfter=8)
    style_cap   = ParagraphStyle("C", parent=styles["Normal"], fontSize=8,
                                 textColor=colors.HexColor("#7f8c8d"),
                                 alignment=TA_CENTER, spaceAfter=10)

    doc   = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4,
                              leftMargin=2*rl_cm, rightMargin=2*rl_cm,
                              topMargin=2*rl_cm,  bottomMargin=2*rl_cm)
    story = []

    # Title
    story.append(Paragraph("Chest X-Ray Pneumonia Detection", style_title))
    story.append(Paragraph("AI Model Performance Report", style_sub))
    story.append(Paragraph(
        f"Generated on {date.today().strftime('%B %d, %Y')}  ·  "
        f"Model: DenseNet121 (fine-tuned)  ·  Dataset: chest-xray-pneumonia (Kaggle)",
        style_sub))
    story.append(HRFlowable(width="100%", thickness=1.5, color=LIGHT_BLUE, spaceAfter=12))

    # 1. Overview
    story.append(Paragraph("1. Project Overview", style_sec))
    story.append(Paragraph(
        "This report summarises the performance of a deep learning model trained to classify "
        "chest X-ray images as either <b>NORMAL</b> or <b>PNEUMONIA</b>. "
        "The model uses a pretrained <b>DenseNet121</b> architecture (pre-trained on ImageNet) "
        "with a custom classifier head, fine-tuned on the Kaggle chest-xray-pneumonia dataset. "
        "Training metrics were tracked with <b>MLflow</b>. "
        "Grad-CAM heatmaps explain which regions the model focused on for each decision.",
        style_body))

    # 2. Metrics table
    story.append(Paragraph("2. Accuracy & Metrics", style_sec))
    table_data = [
        ["Metric",             "Result"],
        ["Overall Accuracy",   f"{metrics['overall_acc']*100:.2f}%"],
        ["Precision",          f"{metrics['precision']:.3f}"],
        ["Recall",             f"{metrics['recall']:.3f}"],
        ["F1-Score",           f"{metrics['f1']:.3f}"],
        ["ROC AUC",            f"{metrics['roc_auc']:.3f}"],
        ["NORMAL Accuracy",    f"{metrics['class_acc']['NORMAL']*100:.2f}%"],
        ["PNEUMONIA Accuracy", f"{metrics['class_acc']['PNEUMONIA']*100:.2f}%"],
        ["Total Test Images",  str(metrics["dataset_sizes"].get("test", "?"))],
        ["Architecture",       "DenseNet121 (pretrained on ImageNet)"],
    ]
    tbl = Table(table_data, colWidths=[9*rl_cm, 7*rl_cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 11),
        ("ALIGN",        (0, 0), (-1, 0), "CENTER"),
        ("BACKGROUND",   (0, 1), (-1, 1), LIGHT_GREY),
        ("BACKGROUND",   (0, 3), (-1, 3), LIGHT_GREY),
        ("BACKGROUND",   (0, 5), (-1, 5), LIGHT_GREY),
        ("BACKGROUND",   (0, 7), (-1, 7), LIGHT_GREY),
        ("BACKGROUND",   (0, 9), (-1, 9), LIGHT_GREY),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 10),
        ("ALIGN",        (1, 1), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",         (0, 0), (-1, -1), 0.5, MID_GREY),
        ("TOPPADDING",   (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # 3. Summary
    story.append(Paragraph("3. Performance Summary", style_sec))
    story.append(Paragraph(summary_text, style_body))

    # 4. Confusion matrix + ROC side by side
    story.append(Paragraph("4. Confusion Matrix & ROC Curve", style_sec))
    story.append(Paragraph(
        "The <b>confusion matrix</b> shows correct vs incorrect predictions per class. "
        "The <b>ROC curve</b> shows the model's ability to distinguish the two classes "
        "across all confidence thresholds — AUC closer to 1.0 is better.",
        style_body))

    cm_img  = RLImage(cm_buf,  width=8*rl_cm, height=6.4*rl_cm)
    roc_img = RLImage(roc_buf, width=8*rl_cm, height=6.7*rl_cm)
    side_by_side = Table([[cm_img, roc_img]], colWidths=[8.5*rl_cm, 8.5*rl_cm])
    side_by_side.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                       ("VALIGN",(0, 0), (-1, -1), "MIDDLE")]))
    story.append(side_by_side)
    story.append(Paragraph(
        "Figure 1 — Left: Confusion matrix (rows = true labels, cols = predicted). "
        "Right: ROC curve with AUC score.", style_cap))

    # 5. Grad-CAM
    story.append(Paragraph("5. Grad-CAM Heatmap Samples", style_sec))
    story.append(Paragraph(
        "Grad-CAM highlights the regions of each X-ray that most influenced the model's "
        "decision. <b>Red/orange areas</b> indicate high focus; <b>blue areas</b> were "
        "largely ignored. PNEUMONIA cases typically show focus on opaque/consolidated "
        "lung regions; NORMAL cases show diffuse attention across clear lung fields.",
        style_body))
    gc_w  = 13*rl_cm
    gc_h  = gc_w * (NUM_GRADCAM * 3.6) / (2 * 9)
    story.append(RLImage(gc_buf, width=gc_w, height=gc_h))
    story.append(Paragraph(
        "Figure 2 — Left: original X-ray. Right: Grad-CAM overlay. "
        "Green border = correct, red border = incorrect prediction.", style_cap))

    # 6. Limitations
    story.append(Paragraph("6. Limitations & Next Steps", style_sec))
    story.append(Paragraph(
        "This model is for <b>educational purposes only</b> and must not be used for "
        "clinical diagnosis. Limitations: (1) Kaggle validation set has only 16 images, "
        "making validation accuracy noisy. (2) Training data comes from a single source "
        "and may not generalise to different equipment or populations. "
        "(3) Binary classification only — real tools cover many more conditions. "
        "Next steps: unfreeze deeper DenseNet layers, add class-weighted loss to handle "
        "imbalance, or fine-tune on a larger multi-institution dataset.",
        style_body))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.8, color=MID_GREY))
    story.append(Paragraph(
        "Chest X-Ray Pneumonia Detection  ·  PyTorch + DenseNet121  ·  Educational use only",
        style_cap))

    doc.build(story)
    print(f"\n{'='*52}")
    print(f"  Report saved to: {OUTPUT_PDF}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    generate_report()
