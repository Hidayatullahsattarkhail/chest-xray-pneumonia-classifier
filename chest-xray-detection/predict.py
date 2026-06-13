"""
predict.py — Run inference on a chest X-ray image.

Usage:
    python predict.py path/to/xray.jpg

Or import and use the predict() function directly:
    from predict import predict
    label, confidence = predict("xray.jpg")
    print(label, confidence)
"""

import sys
import torch
from PIL import Image

from model import load_model
from dataset import get_transforms, CLASS_NAMES

# ── Configuration ─────────────────────────────────────────────────────────────

from config import CHECKPOINT

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Inference function ────────────────────────────────────────────────────────

def predict(image_path: str, checkpoint: str = CHECKPOINT) -> tuple[str, float]:
    """
    Predict whether a chest X-ray shows NORMAL lungs or PNEUMONIA.

    Args:
        image_path: Path to the X-ray image file (JPG, PNG, etc.).
        checkpoint: Path to the trained model weights (.pth file).

    Returns:
        A tuple of (label, confidence):
            label      — "NORMAL" or "PNEUMONIA"
            confidence — Float between 0 and 1 (e.g. 0.97 means 97% confident)

    Example:
        label, confidence = predict("patient_xray.jpg")
        print(f"Result: {label} ({confidence * 100:.1f}% confidence)")
    """
    # Load model
    model = load_model(checkpoint, device)

    # Load and preprocess image
    transform = get_transforms("test")   # No augmentation for inference
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        outputs     = model(input_tensor)          # Raw scores
        probs       = torch.softmax(outputs, dim=1)  # Convert to probabilities
        pred_idx    = probs.argmax(dim=1).item()
        confidence  = probs[0][pred_idx].item()

    label = CLASS_NAMES[pred_idx]
    return label, confidence


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_xray_image>")
        print("Example: python predict.py data/test/NORMAL/IM-0001-0001.jpeg")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"\nAnalysing: {image_path}")
    label, confidence = predict(image_path)

    print("\n" + "=" * 40)
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence * 100:.1f}%")
    print("=" * 40 + "\n")

    if label == "PNEUMONIA":
        print("⚠️  Pneumonia indicators detected.")
    else:
        print("✓  No pneumonia indicators detected.")


if __name__ == "__main__":
    main()
