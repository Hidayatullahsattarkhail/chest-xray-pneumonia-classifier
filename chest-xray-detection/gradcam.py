"""
gradcam.py — Grad-CAM heatmap visualization using the pytorch-grad-cam library.

Grad-CAM answers the question:
  "Which parts of the X-ray did the model look at to make its decision?"

It draws a heatmap on top of the original image:
  🔴 Red / hot   → the model focused here (most important regions)
  🔵 Blue / cool → the model mostly ignored these regions

Usage:
    python gradcam.py path/to/xray.jpg

Or import and use directly:
    from gradcam import generate_heatmap, run_gradcam
    result = generate_heatmap(model, "xray.jpg", device)
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import load_model, get_gradcam_target_layer
from dataset import get_transforms, CLASS_NAMES


# ── Configuration ─────────────────────────────────────────────────────────────

CHECKPOINT = "best_model.pth"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Core helper: generate heatmap for one image ───────────────────────────────

def generate_heatmap(model, image_path: str, device) -> dict:
    """
    Run prediction and generate a Grad-CAM heatmap for one X-ray image.

    Uses the pytorch-grad-cam library, which handles hooks and gradient
    computation automatically.

    Args:
        model:      Loaded PyTorch model in eval mode.
        image_path: Path to the X-ray image file.
        device:     torch.device for computation.

    Returns:
        A dict with:
            original   — PIL Image (original X-ray, 224x224)
            overlaid   — numpy array uint8 (heatmap blended on X-ray)
            pred_label — "NORMAL" or "PNEUMONIA"
            confidence — float, e.g. 0.97
            true_label — inferred from folder name, or None
    """
    transform = get_transforms("test")

    # Load image
    original_pil = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_pil).unsqueeze(0).to(device)

    # Prediction
    model.eval()
    with torch.no_grad():
        output     = model(input_tensor)
        probs      = torch.softmax(output, dim=1)
        pred_idx   = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()
        pred_label = CLASS_NAMES[pred_idx]

    # Grad-CAM via library
    target_layer = get_gradcam_target_layer(model)
    targets      = [ClassifierOutputTarget(pred_idx)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        # cam() returns a (1, H, W) numpy array, values in [0, 1]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Prepare float RGB image in [0, 1] for the library's overlay function
    img_resized = original_pil.resize((224, 224)).convert("RGB")
    img_float   = np.array(img_resized, dtype=np.float32) / 255.0

    # show_cam_on_image blends the heatmap onto the image — returns uint8 RGB
    overlaid = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # Try to infer true label from the folder name
    true_label = None
    for name in CLASS_NAMES:
        if f"/{name}/" in image_path or f"\\{name}\\" in image_path:
            true_label = name
            break

    return {
        "original":   original_pil.resize((224, 224)),
        "overlaid":   overlaid,        # numpy uint8 (H, W, 3)
        "pred_label": pred_label,
        "confidence": confidence,
        "true_label": true_label,
        "path":       image_path,
    }


# ── Main single-image function ────────────────────────────────────────────────

def run_gradcam(image_path: str, checkpoint: str = CHECKPOINT, save: bool = True):
    """
    Generate and display a Grad-CAM heatmap for a chest X-ray image.

    Produces a three-panel figure:
      Left  — original X-ray
      Middle — Grad-CAM heatmap
      Right  — heatmap overlaid on X-ray

    Args:
        image_path: Path to the X-ray image.
        checkpoint: Path to the trained model weights.
        save:       If True, saves the result as a PNG file.
    """
    model  = load_model(checkpoint, device)
    result = generate_heatmap(model, image_path, device)

    pred_label = result["pred_label"]
    confidence = result["confidence"]
    overlaid   = result["overlaid"]
    original   = result["original"]

    result_color = "#2ecc71" if pred_label == "NORMAL" else "#e74c3c"

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Grad-CAM Analysis   |   Prediction: {pred_label}   |   Confidence: {confidence * 100:.1f}%",
        fontsize=13, fontweight="bold", color=result_color,
    )

    # Original
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original X-Ray", fontsize=11)
    axes[0].axis("off")

    # Heatmap only (extract from overlay by showing grayscale cam)
    axes[1].imshow(overlaid)
    axes[1].set_title("Grad-CAM Heatmap\n(Red = model focused here)", fontsize=11)
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(overlaid)
    axes[2].set_title("Heatmap Overlaid on X-Ray", fontsize=11)
    axes[2].axis("off")

    fig.text(
        0.5, -0.04,
        "🔴 Red / hot = where the model focused most  |  🔵 Blue / cool = less important regions",
        ha="center", fontsize=10, style="italic", color="#555555",
    )
    plt.tight_layout()

    if save:
        basename  = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f"gradcam_{basename}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    plt.show()

    print("\n" + "=" * 50)
    print(f"  Prediction : {pred_label}")
    print(f"  Confidence : {confidence * 100:.1f}%")
    print("=" * 50)
    print("\nHow to read the heatmap:")
    print("  🔴 Red/orange areas — the model focused most here")
    print("  🟡 Yellow areas     — moderately important")
    print("  🔵 Blue areas       — mostly ignored by the model")
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gradcam.py <path_to_xray_image>")
        print("Example: python gradcam.py data/test/PNEUMONIA/person1_virus_1.jpeg")
        sys.exit(1)

    run_gradcam(image_path=sys.argv[1])
