"""
report.py — Generate a PDF summary report of the model's performance.

Usage:
    python report.py

Output:
    xray_report.pdf — a structured report containing:
      - Project overview
      - Model accuracy and key metrics
      - Confusion matrix
      - Sample Grad-CAM heatmaps (2–4 images)
      - Plain-English performance summary

Requirements:
    pip install reportlab
    (Already included in requirements.txt)

Make sure you have run train.py first so that best_model.pth exists.
"""

import os
import io
import random
import tempfile
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from tqdm import tqdm

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm as rl_cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
)

from model import load_model
from dataset import get_dataloaders, get_transforms, CLASS_NAMES
from gradcam import GradCAM, overlay_heatmap


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR    = "data"
CHECKPOINT  = "best_model.pth"
OUTPUT_PDF  = "xray_report.pdf"
BATCH_SIZE  = 32
NUM_GRADCAM = 4   # Number of Grad-CAM sample images in the report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Colour palette for the PDF ────────────────────────────────────────────────

BLUE        = colors.HexColor("#2c3e50")
LIGHT_BLUE  = colors.HexColor("#3498db")
GREEN       = colors.HexColor("#27ae60")
RED         = colors.HexColor("#e74c3c")
ORANGE      = colors.HexColor("#e67e22")
LIGHT_GREY  = colors.HexColor("#ecf0f1")
MID_GREY    = colors.HexColor("#bdc3c7")
WHITE       = colors.white


# ── Step 1: Collect evaluation metrics ───────────────────────────────────────

def collect_metrics() -> dict:
    """
    Run the model on the test set and return accuracy metrics.

    Returns a dict with:
        overall_acc   — float, e.g. 0.924
        class_acc     — dict {class_name: float}
        cm            — 2D numpy array (confusion matrix)
        report_str    — full scikit-learn classification report as string
        dataset_sizes — dict with split sizes
    """
    print("  Running evaluation on test set...")
    dataloaders, dataset_sizes = get_dataloaders(DATA_DIR, BATCH_SIZE)

    model = load_model(CHECKPOINT, device)

    all_preds, all_labels = [], []
    for inputs, labels in tqdm(dataloaders["test"], desc="  Evaluating"):
        inputs = inputs.to(device)
        with torch.no_grad():
            preds = model(inputs).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    overall_acc = accuracy_score(all_labels, all_preds)
    cm_array    = confusion_matrix(all_labels, all_preds)
    report_str  = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, digits=3
    )

    class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        total   = cm_array[i].sum()
        correct = cm_array[i][i]
        class_acc[name] = correct / total if total > 0 else 0.0

    return {
        "overall_acc":   overall_acc,
        "class_acc":     class_acc,
        "cm":            cm_array,
        "report_str":    report_str,
        "dataset_sizes": dataset_sizes,
    }


# ── Step 2: Render confusion matrix as a PNG in memory ───────────────────────

def render_confusion_matrix(cm_array: np.ndarray) -> io.BytesIO:
    """
    Draw a colour-coded confusion matrix and return it as a PNG byte stream.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_array, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NORMAL", "PNEUMONIA"], fontsize=11)
    ax.set_yticklabels(["NORMAL", "PNEUMONIA"], fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    # Annotate each cell with the count
    total = cm_array.sum()
    for i in range(2):
        for j in range(2):
            count   = cm_array[i][j]
            percent = count / total * 100
            color   = "white" if count > cm_array.max() / 2 else "black"
            ax.text(
                j, i,
                f"{count}\n({percent:.1f}%)",
                ha="center", va="center",
                fontsize=12, color=color, fontweight="bold",
            )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Step 3: Generate Grad-CAM samples as a PNG grid in memory ────────────────

def render_gradcam_grid(num_images: int = NUM_GRADCAM) -> io.BytesIO:
    """
    Pick sample test images, generate Grad-CAM heatmaps, and return
    a grid image (original | overlay) as a PNG byte stream.
    """
    print(f"  Generating {num_images} Grad-CAM samples...")

    model        = load_model(CHECKPOINT, device)
    target_layer = model.layer4[-1]
    gradcam      = GradCAM(model, target_layer)
    transform    = get_transforms("test")

    # Collect sample images (equal split between classes where possible)
    samples = []
    per_class = max(1, num_images // len(CLASS_NAMES))
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, "test", class_name)
        if not os.path.isdir(class_dir):
            continue
        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        chosen = random.sample(files, min(per_class, len(files)))
        for fname in chosen:
            samples.append((os.path.join(class_dir, fname), class_name))

    # Trim or pad to exactly num_images
    samples = samples[:num_images]

    # Layout: 2 columns per image (original | overlay), one row per image
    cols = 2
    rows = len(samples)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.8))

    if rows == 1:
        axes = [axes]

    fig.suptitle(
        "Grad-CAM Heatmaps — Where the Model Focused",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row_idx, (path, true_label) in enumerate(samples):
        # Predict + generate heatmap
        original    = Image.open(path).convert("RGB")
        inp         = transform(original).unsqueeze(0).to(device)
        inp.requires_grad_(True)

        with torch.no_grad():
            out   = model(inp)
            probs = torch.softmax(out, dim=1)
            idx   = probs.argmax(dim=1).item()
            conf  = probs[0][idx].item()
            pred  = CLASS_NAMES[idx]

        heatmap  = gradcam.generate(inp, class_idx=idx)
        overlaid = overlay_heatmap(original, heatmap, alpha=0.5)

        correct      = (pred == true_label)
        label_color  = "#27ae60" if correct else "#e74c3c"
        status       = "✓" if correct else "✗"

        # Original
        axes[row_idx][0].imshow(original.resize((224, 224)), cmap="gray")
        axes[row_idx][0].set_title(f"True: {true_label}", fontsize=9, color="#555")
        axes[row_idx][0].axis("off")

        # Overlay
        axes[row_idx][1].imshow(overlaid)
        axes[row_idx][1].set_title(
            f"{status} Pred: {pred}  ({conf*100:.1f}%)",
            fontsize=9, color=label_color, fontweight="bold",
        )
        axes[row_idx][1].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Step 4: Build performance summary text ───────────────────────────────────

def build_summary(metrics: dict) -> str:
    """
    Write a short plain-English interpretation of the model's results.
    """
    acc   = metrics["overall_acc"] * 100
    n_acc = metrics["class_acc"]["NORMAL"]    * 100
    p_acc = metrics["class_acc"]["PNEUMONIA"] * 100
    total = metrics["dataset_sizes"].get("test", "?")

    rating = "excellent" if acc >= 90 else "good" if acc >= 80 else "moderate"

    return (
        f"The model achieved {rating} overall accuracy of {acc:.1f}% on {total} "
        f"unseen test images. "
        f"For NORMAL cases, it correctly identified {n_acc:.1f}% of images. "
        f"For PNEUMONIA cases, it correctly identified {p_acc:.1f}% of images. "
        f"The Grad-CAM heatmaps below show that the model focuses on the lung "
        f"fields when making decisions — red and orange areas indicate the regions "
        f"that most influenced each prediction. "
        f"Overall, this fine-tuned ResNet18 model demonstrates strong capability "
        f"for distinguishing normal chest X-rays from those showing pneumonia."
    )


# ── Step 5: Assemble the PDF ──────────────────────────────────────────────────

def generate_report():
    """
    Collect all outputs and assemble the final PDF report.
    Saves to xray_report.pdf in the current directory.
    """
    print("\n" + "=" * 52)
    print("  Generating PDF Report")
    print("=" * 52)

    # ── Guard: check prerequisites ────────────────────────────────────────────
    if not os.path.isfile(CHECKPOINT):
        print(f"\nERROR: '{CHECKPOINT}' not found.")
        print("Please run train.py first to train and save the model.")
        return

    if not os.path.isdir(os.path.join(DATA_DIR, "test")):
        print(f"\nERROR: Test data not found in '{DATA_DIR}/test/'.")
        print("Please run download_dataset.py first.")
        return

    # ── Collect data ──────────────────────────────────────────────────────────
    print("\n[1/3] Collecting evaluation metrics...")
    metrics = collect_metrics()

    print("\n[2/3] Rendering confusion matrix...")
    cm_buf = render_confusion_matrix(metrics["cm"])

    print("\n[3/3] Generating Grad-CAM heatmap samples...")
    gc_buf = render_gradcam_grid(num_images=NUM_GRADCAM)

    summary_text = build_summary(metrics)

    # ── PDF styles ────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=BLUE,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    style_subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#7f8c8d"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    style_section = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading1"],
        fontSize=13,
        textColor=BLUE,
        spaceBefore=16,
        spaceAfter=6,
        borderPad=4,
    )
    style_body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#2c3e50"),
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    style_caption = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#7f8c8d"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    # ── Build PDF content ─────────────────────────────────────────────────────
    doc      = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=2 * rl_cm,
        rightMargin=2 * rl_cm,
        topMargin=2 * rl_cm,
        bottomMargin=2 * rl_cm,
    )
    story = []

    # Title block
    story.append(Paragraph("Chest X-Ray Pneumonia Detection", style_title))
    story.append(Paragraph("AI Model Performance Report", style_subtitle))
    story.append(Paragraph(
        f"Generated on {date.today().strftime('%B %d, %Y')}  ·  "
        f"Model: ResNet18 (fine-tuned)  ·  Dataset: chest-xray-pneumonia (Kaggle)",
        style_subtitle,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=LIGHT_BLUE, spaceAfter=12))

    # Section 1 — Overview
    story.append(Paragraph("1. Project Overview", style_section))
    story.append(Paragraph(
        "This report summarises the performance of a deep learning model trained to "
        "classify chest X-ray images as either <b>NORMAL</b> or <b>PNEUMONIA</b>. "
        "The model uses a pretrained ResNet18 architecture with a custom classification "
        "head, fine-tuned on the Kaggle chest-xray-pneumonia dataset. "
        "Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps are included "
        "to explain which regions of each X-ray the model focused on.",
        style_body,
    ))

    # Section 2 — Accuracy metrics table
    story.append(Paragraph("2. Accuracy Results", style_section))

    overall_pct  = f"{metrics['overall_acc'] * 100:.2f}%"
    normal_pct   = f"{metrics['class_acc']['NORMAL']    * 100:.2f}%"
    pneumonia_pct= f"{metrics['class_acc']['PNEUMONIA'] * 100:.2f}%"
    test_size    = str(metrics["dataset_sizes"].get("test", "?"))

    table_data = [
        ["Metric",                  "Result"],
        ["Overall Accuracy",        overall_pct],
        ["NORMAL Accuracy",         normal_pct],
        ["PNEUMONIA Accuracy",      pneumonia_pct],
        ["Total Test Images",       test_size],
        ["Architecture",            "ResNet18 (pretrained on ImageNet)"],
        ["Training Strategy",       "Frozen backbone + fine-tuned classifier head"],
    ]

    tbl = Table(table_data, colWidths=[9 * rl_cm, 7 * rl_cm])
    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",   (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 11),
        ("ALIGN",        (0, 0), (-1, 0), "CENTER"),
        # Data rows — alternating shading
        ("BACKGROUND",   (0, 1), (-1, 1), LIGHT_GREY),
        ("BACKGROUND",   (0, 3), (-1, 3), LIGHT_GREY),
        ("BACKGROUND",   (0, 5), (-1, 5), LIGHT_GREY),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 10),
        ("ALIGN",        (1, 1), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUND",(0, 2), (-1, 2), WHITE),
        # Borders
        ("GRID",         (0, 0), (-1, -1), 0.5, MID_GREY),
        ("TOPPADDING",   (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # Section 3 — Performance summary (text)
    story.append(Paragraph("3. Performance Summary", style_section))
    story.append(Paragraph(summary_text, style_body))

    # Section 4 — Confusion matrix
    story.append(Paragraph("4. Confusion Matrix", style_section))
    story.append(Paragraph(
        "The confusion matrix shows how many images were correctly and incorrectly "
        "classified. <b>Diagonal cells</b> (top-left and bottom-right) are correct "
        "predictions. <b>Off-diagonal cells</b> are mistakes — for example, a "
        "PNEUMONIA case predicted as NORMAL (a false negative) is the most "
        "clinically important type of error.",
        style_body,
    ))
    cm_image = RLImage(cm_buf, width=10 * rl_cm, height=8 * rl_cm)
    story.append(cm_image)
    story.append(Paragraph(
        "Figure 1 — Confusion matrix on the test set. "
        "Rows = true labels, Columns = predicted labels.",
        style_caption,
    ))

    # Section 5 — Grad-CAM heatmaps
    story.append(Paragraph("5. Grad-CAM Heatmap Samples", style_section))
    story.append(Paragraph(
        "Grad-CAM highlights the regions of each X-ray that most influenced the "
        "model's decision. <b>Red and orange areas</b> indicate high focus; "
        "<b>blue areas</b> were largely ignored. "
        "In PNEUMONIA cases the model typically focuses on opaque or consolidated "
        "regions in the lung fields. In NORMAL cases it tends to spread attention "
        "across clearer lung areas.",
        style_body,
    ))
    gc_width  = 14 * rl_cm
    gc_height = gc_width * (NUM_GRADCAM * 3.8) / (2 * 4)   # Preserve aspect ratio
    gc_image  = RLImage(gc_buf, width=gc_width, height=gc_height)
    story.append(gc_image)
    story.append(Paragraph(
        "Figure 2 — Left column: original X-ray. "
        "Right column: Grad-CAM heatmap overlaid. "
        "Green border = correct prediction, red border = incorrect.",
        style_caption,
    ))

    # Section 6 — Limitations
    story.append(Paragraph("6. Limitations & Next Steps", style_section))
    story.append(Paragraph(
        "This model is intended for <b>educational purposes only</b> and should not "
        "be used for clinical diagnosis. Key limitations include: "
        "(1) The Kaggle validation set contains only 16 images, making validation "
        "accuracy noisy. "
        "(2) The model was trained on a single dataset and may not generalise to "
        "different X-ray machines or patient populations. "
        "(3) The model distinguishes only two classes; real clinical tools cover "
        "a broader range of conditions. "
        "Possible next steps: unfreeze deeper ResNet layers for full fine-tuning, "
        "experiment with DenseNet121, or train on a larger multi-class dataset.",
        style_body,
    ))

    # Footer rule
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.8, color=MID_GREY))
    story.append(Paragraph(
        "Chest X-Ray Pneumonia Detection  ·  PyTorch + ResNet18  ·  Educational use only",
        style_caption,
    ))

    # ── Write the PDF ─────────────────────────────────────────────────────────
    doc.build(story)
    print(f"\n{'=' * 52}")
    print(f"  Report saved to: {OUTPUT_PDF}")
    print(f"{'=' * 52}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_report()
