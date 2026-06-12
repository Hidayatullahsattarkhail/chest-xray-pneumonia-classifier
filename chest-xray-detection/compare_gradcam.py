"""
compare_gradcam.py — Side-by-side Grad-CAM comparison for multiple X-rays.

Shows a grid where each row is one image:
  [ Original X-Ray ]  [ Grad-CAM Overlay ]

Useful for seeing how the model's focus differs between NORMAL and PNEUMONIA.

Usage:
    python compare_gradcam.py                       # auto-pick 2 per class
    python compare_gradcam.py --num-per-class 3     # 3 per class
    python compare_gradcam.py img1.jpg img2.jpg     # specific images
"""

import os
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from model import load_model
from dataset import CLASS_NAMES
from gradcam import generate_heatmap


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = "data/test"
CHECKPOINT = "best_model.pth"

CORRECT_COLOR   = "#2ecc71"
WRONG_COLOR     = "#e74c3c"
NORMAL_COLOR    = "#3498db"
PNEUMONIA_COLOR = "#e67e22"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Auto-pick images from the test folder ─────────────────────────────────────

def pick_images_from_test(data_dir: str, num_per_class: int = 2) -> list:
    selected = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: folder not found — {class_dir}")
            continue
        files  = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        chosen = random.sample(files, min(num_per_class, len(files)))
        for fname in chosen:
            selected.append((os.path.join(class_dir, fname), class_name))
    return selected


# ── Build the comparison figure ───────────────────────────────────────────────

def build_figure(results: list) -> plt.Figure:
    num_rows = len(results)
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, num_rows * 4.2))

    if num_rows == 1:
        axes = [axes]

    fig.suptitle(
        "Grad-CAM Comparison — NORMAL vs PNEUMONIA\n"
        "Red/hot areas show where the model focused most",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for row_idx, r in enumerate(results):
        ax_orig    = axes[row_idx][0]
        ax_overlay = axes[row_idx][1]

        is_correct   = (r["true_label"] is None or r["pred_label"] == r["true_label"])
        border_color = CORRECT_COLOR if is_correct else WRONG_COLOR
        class_color  = NORMAL_COLOR  if r["pred_label"] == "NORMAL" else PNEUMONIA_COLOR
        status       = "✓" if is_correct else "✗"

        # Original
        ax_orig.imshow(r["original"], cmap="gray")
        ax_orig.axis("off")
        true_str = f"True: {r['true_label']}" if r["true_label"] else ""
        ax_orig.set_ylabel(true_str, fontsize=10, rotation=0, labelpad=60,
                           va="center", color="#555555")
        ax_orig.set_title("Original X-Ray", fontsize=10, color="#555")
        for spine in ax_orig.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        # Overlay
        ax_overlay.imshow(r["overlaid"])
        ax_overlay.axis("off")
        ax_overlay.set_title(
            f"{status}  Pred: {r['pred_label']}  ({r['confidence']*100:.1f}%)",
            fontsize=10, color=class_color, fontweight="bold",
        )
        for spine in ax_overlay.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    patches = [
        mpatches.Patch(color=CORRECT_COLOR,   label="Correct prediction ✓"),
        mpatches.Patch(color=WRONG_COLOR,     label="Wrong prediction  ✗"),
        mpatches.Patch(color=NORMAL_COLOR,    label="Predicted: NORMAL"),
        mpatches.Patch(color=PNEUMONIA_COLOR, label="Predicted: PNEUMONIA"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    return fig


# ── Main function ─────────────────────────────────────────────────────────────

def compare_gradcam(image_paths=None, num_per_class: int = 2,
                    checkpoint: str = CHECKPOINT, save: bool = True):
    # Decide which images to use
    if image_paths:
        samples = [(p, None) for p in image_paths]
    else:
        samples = pick_images_from_test(DATA_DIR, num_per_class=num_per_class)

    if not samples:
        print(f"ERROR: No images found in '{DATA_DIR}'.")
        return

    # Load model once and reuse for all images
    print(f"Loading model from: {checkpoint}")
    model = load_model(checkpoint, device)

    # Process each image
    results = []
    for path, true_label in samples:
        print(f"  Processing: {os.path.basename(path)}")
        r = generate_heatmap(model, path, device)
        if true_label:
            r["true_label"] = true_label
        results.append(r)

    # Build and display figure
    print(f"\nGenerating comparison grid for {len(results)} images...")
    fig = build_figure(results)

    if save:
        save_path = "compare_gradcam.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison saved to: {save_path}")

    plt.show()

    # Summary
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
    print("\nHow to read: 🔴 Red = focused  |  🔵 Blue = ignored")
    print("NORMAL    → focus often on clear, dark lung fields")
    print("PNEUMONIA → focus often on bright, opaque lung regions")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Grad-CAM heatmaps.")
    parser.add_argument("images", nargs="*", help="Optional: specific image paths.")
    parser.add_argument("--num-per-class", type=int, default=2,
                        help="Images per class when auto-picking (default: 2).")
    args = parser.parse_args()

    compare_gradcam(
        image_paths=args.images if args.images else None,
        num_per_class=args.num_per_class,
    )
