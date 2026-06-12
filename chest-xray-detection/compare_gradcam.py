"""
compare_gradcam.py — Side-by-side Grad-CAM comparison for multiple X-rays.

Shows a grid where each row is one image:
  [ Original X-Ray ]  [ Grad-CAM Heatmap ]

This is useful for seeing how the model's focus differs between
NORMAL lungs and PNEUMONIA cases.

Usage:
    # Auto-pick images from the test folder (recommended)
    python compare_gradcam.py

    # Specify your own images
    python compare_gradcam.py img1.jpg img2.jpg img3.jpg

    # Control how many auto-picked images per class
    python compare_gradcam.py --num-per-class 3
"""

import sys
import os
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image

from model import load_model
from dataset import get_transforms, CLASS_NAMES
from gradcam import GradCAM, overlay_heatmap


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = "data/test"
CHECKPOINT = "best_model.pth"

CORRECT_COLOR = "#2ecc71"   # Green  — correct prediction
WRONG_COLOR   = "#e74c3c"   # Red    — wrong prediction
NORMAL_COLOR  = "#3498db"   # Blue   — section header for NORMAL
PNEUMONIA_COLOR = "#e67e22" # Orange — section header for PNEUMONIA

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Auto-pick images from the test folder ─────────────────────────────────────

def pick_images_from_test(data_dir: str, num_per_class: int = 2) -> list[tuple[str, str]]:
    """
    Randomly select images from each class in the test folder.

    Args:
        data_dir:       Path to the test/ folder.
        num_per_class:  How many images to pick per class.

    Returns:
        List of (image_path, true_label) tuples.
        NORMAL images come first, then PNEUMONIA.
    """
    selected = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: folder not found — {class_dir}")
            continue

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        chosen = random.sample(files, min(num_per_class, len(files)))
        for fname in chosen:
            selected.append((os.path.join(class_dir, fname), class_name))

    return selected


# ── Process one image: predict + generate heatmap ─────────────────────────────

def process_image(model, gradcam, image_path: str) -> dict:
    """
    Run prediction and Grad-CAM for a single image.

    Args:
        model:      Loaded PyTorch model.
        gradcam:    GradCAM instance hooked to the model.
        image_path: Path to the X-ray image.

    Returns:
        A dict with keys:
            original   — PIL Image (original X-ray)
            overlaid   — PIL Image (heatmap blended on X-ray)
            heatmap    — 2D numpy array (raw heatmap)
            pred_label — "NORMAL" or "PNEUMONIA"
            confidence — float, e.g. 0.97
            true_label — true class if derivable from path, else None
    """
    transform    = get_transforms("test")
    original_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # Prediction (without gradient tracking, just for the label)
    with torch.no_grad():
        output     = model(input_tensor)
        probs      = torch.softmax(output, dim=1)
        pred_idx   = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()
        pred_label = CLASS_NAMES[pred_idx]

    # Grad-CAM (needs gradient tracking — re-runs forward+backward)
    heatmap  = gradcam.generate(input_tensor, class_idx=pred_idx)
    overlaid = overlay_heatmap(original_img, heatmap, alpha=0.5)

    # Try to infer the true label from the folder name
    true_label = None
    for name in CLASS_NAMES:
        if f"/{name}/" in image_path or f"\\{name}\\" in image_path:
            true_label = name
            break

    return {
        "original":   original_img,
        "overlaid":   overlaid,
        "heatmap":    heatmap,
        "pred_label": pred_label,
        "confidence": confidence,
        "true_label": true_label,
        "path":       image_path,
    }


# ── Build the comparison figure ───────────────────────────────────────────────

def build_figure(results: list[dict]) -> plt.Figure:
    """
    Build a matplotlib figure with one row per image.

    Each row shows:
      Col 1 — Original X-Ray
      Col 2 — Grad-CAM heatmap overlaid on the X-Ray

    Args:
        results: List of dicts from process_image().

    Returns:
        The completed matplotlib Figure.
    """
    num_images = len(results)
    num_cols   = 2  # original | heatmap overlay
    num_rows   = num_images

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(num_cols * 5, num_rows * 4.5),
    )

    # Ensure axes is always 2D even when there is only one row
    if num_rows == 1:
        axes = [axes]

    fig.suptitle(
        "Grad-CAM Comparison — NORMAL vs PNEUMONIA\n"
        "Red/hot areas show where the model focused most",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for row_idx, r in enumerate(results):
        ax_orig   = axes[row_idx][0]
        ax_heatmap = axes[row_idx][1]

        # ── Determine label colours ───────────────────────────────────────────
        is_correct    = (r["true_label"] is None or r["pred_label"] == r["true_label"])
        border_color  = CORRECT_COLOR if is_correct else WRONG_COLOR
        class_color   = NORMAL_COLOR  if r["pred_label"] == "NORMAL" else PNEUMONIA_COLOR

        # ── Panel 1: Original X-Ray ───────────────────────────────────────────
        ax_orig.imshow(r["original"].resize((224, 224)), cmap="gray")
        ax_orig.axis("off")

        # Coloured border around each image
        for spine in ax_orig.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        # Row label on the left (True label if known)
        true_str = f"True: {r['true_label']}" if r["true_label"] else ""
        ax_orig.set_ylabel(
            true_str,
            fontsize=10,
            rotation=0,
            labelpad=60,
            va="center",
            color="#555555",
        )
        ax_orig.set_title("Original X-Ray", fontsize=10, color="#555555")

        # ── Panel 2: Heatmap Overlay ──────────────────────────────────────────
        ax_heatmap.imshow(r["overlaid"])
        ax_heatmap.axis("off")

        for spine in ax_heatmap.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        # Title: prediction + confidence
        status    = "✓" if is_correct else "✗"
        conf_pct  = r["confidence"] * 100
        title_str = (
            f"{status}  Pred: {r['pred_label']}\n"
            f"Confidence: {conf_pct:.1f}%"
        )
        ax_heatmap.set_title(title_str, fontsize=10, color=class_color, fontweight="bold")

    # ── Legend at the bottom ──────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=CORRECT_COLOR,   label="Correct prediction ✓"),
        mpatches.Patch(color=WRONG_COLOR,     label="Wrong prediction  ✗"),
        mpatches.Patch(color=NORMAL_COLOR,    label="Predicted: NORMAL"),
        mpatches.Patch(color=PNEUMONIA_COLOR, label="Predicted: PNEUMONIA"),
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    return fig


# ── Main function ─────────────────────────────────────────────────────────────

def compare_gradcam(
    image_paths: list[str] = None,
    num_per_class: int = 2,
    checkpoint: str = CHECKPOINT,
    save: bool = True,
):
    """
    Generate and display a Grad-CAM comparison grid for multiple X-ray images.

    Args:
        image_paths:   List of image file paths to compare.
                       If None, images are auto-picked from DATA_DIR.
        num_per_class: When auto-picking, how many images per class.
        checkpoint:    Path to trained model weights.
        save:          If True, save the figure as a PNG.
    """
    # 1. Decide which images to use
    if image_paths:
        # User-provided paths — we don't know the true labels
        samples = [(p, None) for p in image_paths]
    else:
        # Auto-pick from the test folder
        samples = pick_images_from_test(DATA_DIR, num_per_class=num_per_class)

    if not samples:
        print("ERROR: No images to process.")
        print(f"Make sure '{DATA_DIR}' contains NORMAL/ and PNEUMONIA/ subfolders.")
        return

    # 2. Load model and set up Grad-CAM
    print(f"Loading model from: {checkpoint}")
    model        = load_model(checkpoint, device)
    target_layer = model.layer4[-1]   # Last conv block of ResNet18
    gradcam      = GradCAM(model, target_layer)

    # 3. Process each image
    results = []
    for path, true_label in samples:
        print(f"  Processing: {os.path.basename(path)} (true: {true_label or 'unknown'})")
        result = process_image(model, gradcam, path)
        if true_label:
            result["true_label"] = true_label  # Use the known true label
        results.append(result)

    # 4. Build and display the figure
    print(f"\nGenerating comparison grid for {len(results)} images...")
    fig = build_figure(results)

    if save:
        save_path = "compare_gradcam.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison saved to: {save_path}")

    plt.show()

    # 5. Print a plain-English summary
    print("\n── Summary ──────────────────────────────────────────")
    for r in results:
        correct_str = ""
        if r["true_label"]:
            match       = "✓" if r["pred_label"] == r["true_label"] else "✗"
            correct_str = f"  {match}  True: {r['true_label']}"
        print(
            f"  {os.path.basename(r['path']):<40}"
            f"  →  {r['pred_label']:<12} ({r['confidence']*100:.1f}%)"
            f"{correct_str}"
        )

    print()
    print("How to read the heatmap overlay:")
    print("  🔴 Red / hot   → the model focused strongly here")
    print("  🟡 Yellow      → moderate focus")
    print("  🔵 Blue / cool → the model paid little attention here")
    print()
    print("What to look for:")
    print("  NORMAL    — focus often on clear, dark lung fields")
    print("  PNEUMONIA — focus often on bright opaque regions in the lungs")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Grad-CAM heatmaps across multiple X-ray images."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Optional: paths to specific image files to compare.",
    )
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=2,
        help="When auto-picking from test folder: images per class (default: 2).",
    )
    args = parser.parse_args()

    compare_gradcam(
        image_paths=args.images if args.images else None,
        num_per_class=args.num_per_class,
    )
