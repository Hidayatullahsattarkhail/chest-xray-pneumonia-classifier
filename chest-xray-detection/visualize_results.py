"""
visualize_results.py — Display a grid of chest X-ray predictions.

Usage:
    python visualize_results.py              # shows 12 random test images
    python visualize_results.py --num 16     # shows 16 images
    python visualize_results.py --only-wrong # shows only incorrect predictions

Make sure you have already run train.py so that best_model.pth exists.
"""

import argparse
import random
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image

from model import load_model
from dataset import get_transforms, CLASS_NAMES


# ── Configuration ─────────────────────────────────────────────────────────────

from config import CHECKPOINT, TEST_DATA_DIR as DATA_DIR

# Colour coding: green = correct, red = wrong
CORRECT_COLOR = "#2ecc71"   # green
WRONG_COLOR   = "#e74c3c"   # red

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Collect all test image paths with their true labels ───────────────────────

def collect_test_images(data_dir: str) -> list[tuple[str, str]]:
    """
    Walk the test folder and return a list of (image_path, true_label) pairs.

    Args:
        data_dir: Path to test/ folder containing NORMAL/ and PNEUMONIA/ subfolders.

    Returns:
        List of (path, label) tuples, e.g. ("data/test/NORMAL/im001.jpg", "NORMAL")
    """
    samples = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: folder not found — {class_dir}")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(class_dir, fname), class_name))
    return samples


# ── Predict a single image ────────────────────────────────────────────────────

def predict_image(model, image_path: str) -> tuple[str, float]:
    """
    Run inference on one image file.

    Returns:
        (predicted_label, confidence) e.g. ("PNEUMONIA", 0.97)
    """
    transform = get_transforms("test")
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        idx     = probs.argmax(dim=1).item()
        conf    = probs[0][idx].item()

    return CLASS_NAMES[idx], conf


# ── Main visualisation ────────────────────────────────────────────────────────

def visualize(num_images: int = 12, only_wrong: bool = False):
    """
    Display a grid of chest X-ray images with predictions and confidence scores.

    Args:
        num_images: How many images to show in the grid.
        only_wrong: If True, show only images the model got wrong.
    """
    # 1. Load model
    model = load_model(CHECKPOINT, device)

    # 2. Collect test images
    all_samples = collect_test_images(DATA_DIR)
    if not all_samples:
        print(f"ERROR: No test images found in '{DATA_DIR}'.")
        print("Make sure the dataset has been downloaded and the path is correct.")
        return

    # Shuffle so we see a variety of cases each time
    random.shuffle(all_samples)

    # 3. Run predictions and collect results
    results = []
    print(f"Scanning test images...")

    for path, true_label in all_samples:
        pred_label, confidence = predict_image(model, path)
        is_correct = (pred_label == true_label)

        if only_wrong and is_correct:
            continue  # Skip correct predictions if --only-wrong flag is set

        results.append({
            "path":       path,
            "true":       true_label,
            "pred":       pred_label,
            "conf":       confidence,
            "correct":    is_correct,
        })

        if len(results) >= num_images:
            break

    if not results:
        print("No images found matching the filter. Try without --only-wrong.")
        return

    # 4. Build the grid layout
    cols = 4
    rows = (len(results) + cols - 1) // cols   # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    fig.suptitle(
        "Chest X-Ray Predictions — Green = Correct  |  Red = Wrong",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for i, ax in enumerate(axes):
        if i < len(results):
            r = results[i]

            # Show the X-ray image (grayscale look)
            img = Image.open(r["path"]).convert("RGB")
            ax.imshow(img, cmap="gray")

            # Choose border colour based on correctness
            border_color = CORRECT_COLOR if r["correct"] else WRONG_COLOR
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4)

            # Title: prediction + confidence
            status = "✓" if r["correct"] else "✗"
            title = (
                f"{status} Pred: {r['pred']}\n"
                f"Confidence: {r['conf'] * 100:.1f}%\n"
                f"True: {r['true']}"
            )
            title_color = CORRECT_COLOR if r["correct"] else WRONG_COLOR
            ax.set_title(title, fontsize=9, color=title_color, fontweight="bold")
            ax.axis("off")
        else:
            # Hide empty subplot slots
            ax.axis("off")

    # 5. Add a legend at the bottom
    correct_patch = mpatches.Patch(color=CORRECT_COLOR, label="Correct prediction ✓")
    wrong_patch   = mpatches.Patch(color=WRONG_COLOR,   label="Wrong prediction  ✗")
    fig.legend(
        handles=[correct_patch, wrong_patch],
        loc="lower center",
        ncol=2,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()

    # 6. Save the figure to a file
    mode_tag  = "wrong_only" if only_wrong else "sample"
    save_path = f"predictions_{mode_tag}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGrid saved to: {save_path}")

    # 7. Print a quick summary
    num_correct = sum(1 for r in results if r["correct"])
    print(f"Showing {len(results)} images — {num_correct} correct, {len(results) - num_correct} wrong")

    plt.show()
    print("Close the plot window to exit.")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise chest X-ray predictions in a grid."
    )
    parser.add_argument(
        "--num",
        type=int,
        default=12,
        help="Number of images to display (default: 12)",
    )
    parser.add_argument(
        "--only-wrong",
        action="store_true",
        help="Show only images the model predicted incorrectly",
    )
    args = parser.parse_args()

    visualize(num_images=args.num, only_wrong=args.only_wrong)
