"""
gradcam.py — Grad-CAM heatmap visualization for chest X-ray predictions.

Grad-CAM answers the question:
  "Which parts of the X-ray did the model look at to make its decision?"

It draws a heatmap on top of the original image:
  🔴 Red/hot areas  → the model focused here (most important regions)
  🔵 Blue/cool areas → the model mostly ignored these regions

Usage:
    python gradcam.py path/to/xray.jpg
    python gradcam.py data/test/PNEUMONIA/person1_virus_1.jpeg

Or use the function directly in your own code:
    from gradcam import run_gradcam
    run_gradcam("my_xray.jpg")
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from model import load_model
from dataset import get_transforms, CLASS_NAMES


# ── Configuration ─────────────────────────────────────────────────────────────

CHECKPOINT = "best_model.pth"

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Grad-CAM Class ────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping).

    How it works — step by step:
      1. Run the image through the model (forward pass).
      2. Look at the last convolutional layer — it has learned the most
         meaningful visual features.
      3. Run a backward pass to find out how much each feature map
         contributed to the final prediction (gradients).
      4. Average those gradients across the feature map (global average pool).
      5. Weight each feature map by its importance score.
      6. Sum them all up and apply ReLU (keep only positive influences).
      7. Resize the result to the original image size → that is the heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model:        The trained PyTorch model.
            target_layer: The convolutional layer to hook into.
                          For ResNet18, this is model.layer4[-1].
        """
        self.model        = model
        self.target_layer = target_layer

        # These will be filled in by the hooks during forward/backward pass
        self._feature_maps = None  # Output of the target conv layer
        self._gradients    = None  # Gradients flowing back through that layer

        # Register hooks — these run automatically during forward/backward passes
        self._register_hooks()

    def _register_hooks(self):
        """
        Attach two hooks to the target layer:
          - forward hook  → captures the layer's output (feature maps)
          - backward hook → captures the gradients at that layer
        """
        def forward_hook(module, input, output):
            # 'output' is the feature map tensor produced by this layer
            self._feature_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # 'grad_output[0]' is the gradient of the loss w.r.t. this layer's output
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given image tensor.

        Args:
            input_tensor: Preprocessed image tensor, shape (1, 3, H, W).
            class_idx:    Class to explain. If None, uses the predicted class.

        Returns:
            heatmap: A 2D numpy array (H, W) with values in [0, 1].
                     Higher values mean "more important" regions.
        """
        self.model.eval()

        # --- Step 1: Forward pass ---
        # We need gradients, so do NOT use torch.no_grad() here
        output = self.model(input_tensor)  # shape: (1, num_classes)

        # If no class specified, use the top predicted class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # --- Step 2: Backward pass for the chosen class ---
        self.model.zero_grad()

        # Create a one-hot vector to isolate the score for our target class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0

        # Backpropagate — this fills in self._gradients via the hook
        output.backward(gradient=one_hot)

        # --- Step 3: Compute importance weights ---
        # Average the gradients across the spatial dimensions (H, W)
        # Shape: (num_channels,)
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # --- Step 4: Weighted combination of feature maps ---
        # Multiply each feature map by its importance weight, then sum
        cam = (weights * self._feature_maps).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # --- Step 5: ReLU — only keep positive activations ---
        # Negative values mean "evidence against this class" — we ignore them
        cam = F.relu(cam)

        # --- Step 6: Upsample to the original image size (224x224) ---
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        # --- Step 7: Normalise to [0, 1] for display ---
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


# ── Helper: overlay heatmap on image ─────────────────────────────────────────

def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5):
    """
    Blend a Grad-CAM heatmap on top of the original X-ray image.

    Args:
        original_image: PIL Image of the X-ray (any size).
        heatmap:        2D numpy array with values in [0, 1] (from GradCAM.generate).
        alpha:          Heatmap transparency (0 = invisible, 1 = fully opaque).

    Returns:
        A PIL Image combining the original X-ray and the heatmap overlay.
    """
    # Resize original image to 224x224 to match the heatmap
    img_resized = original_image.resize((224, 224)).convert("RGB")
    img_array   = np.array(img_resized) / 255.0  # Normalise to [0, 1]

    # Convert heatmap to RGB using the 'jet' colormap (blue → green → red)
    colormap    = cm.get_cmap("jet")
    heatmap_rgb = colormap(heatmap)[:, :, :3]  # Drop the alpha channel

    # Blend: final = (1 - alpha) * original + alpha * heatmap
    blended = (1 - alpha) * img_array + alpha * heatmap_rgb
    blended = np.clip(blended, 0, 1)

    return Image.fromarray((blended * 255).astype(np.uint8))


# ── Main function ─────────────────────────────────────────────────────────────

def run_gradcam(image_path: str, checkpoint: str = CHECKPOINT, save: bool = True):
    """
    Generate and display a Grad-CAM heatmap for a chest X-ray image.

    Args:
        image_path: Path to the X-ray image file.
        checkpoint: Path to the trained model weights (.pth file).
        save:       If True, saves the result as a PNG file.

    What is shown:
        Left  — original X-ray image
        Middle — Grad-CAM heatmap only (colour scale)
        Right  — heatmap overlaid on the original image
    """
    # 1. Load model
    model = load_model(checkpoint, device)

    # Hook into the last convolutional block of ResNet18
    # layer4[-1] is the deepest layer — it has the most semantic features
    target_layer = model.layer4[-1]
    gradcam      = GradCAM(model, target_layer)

    # 2. Load and preprocess image
    transform    = get_transforms("test")
    original_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)  # Needed for gradient computation

    # 3. Get prediction (before Grad-CAM, just for the label display)
    with torch.no_grad():
        output     = model(input_tensor)
        probs      = torch.softmax(output, dim=1)
        pred_idx   = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()
        pred_label = CLASS_NAMES[pred_idx]

    # 4. Generate heatmap (re-runs forward + backward WITH gradients)
    heatmap  = gradcam.generate(input_tensor, class_idx=pred_idx)
    overlaid = overlay_heatmap(original_img, heatmap, alpha=0.5)

    # 5. Plot the three-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    result_color = "#2ecc71" if pred_label == "NORMAL" else "#e74c3c"

    fig.suptitle(
        f"Grad-CAM Analysis   |   Prediction: {pred_label}   |   Confidence: {confidence * 100:.1f}%",
        fontsize=13,
        fontweight="bold",
        color=result_color,
    )

    # Panel 1 — original X-ray
    axes[0].imshow(original_img.resize((224, 224)), cmap="gray")
    axes[0].set_title("Original X-Ray", fontsize=11)
    axes[0].axis("off")

    # Panel 2 — heatmap only
    heatmap_plot = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap\n(Red = model focused here)", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(heatmap_plot, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — overlay
    axes[2].imshow(overlaid)
    axes[2].set_title("Heatmap Overlaid on X-Ray", fontsize=11)
    axes[2].axis("off")

    # Add a note explaining what the colours mean
    fig.text(
        0.5, -0.04,
        "🔴 Red / hot = where the model focused most  |  🔵 Blue / cool = less important regions",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    plt.tight_layout()

    if save:
        import os
        basename  = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f"gradcam_{basename}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    plt.show()

    # 6. Print a plain-English summary
    print("\n" + "=" * 50)
    print(f"  Prediction : {pred_label}")
    print(f"  Confidence : {confidence * 100:.1f}%")
    print("=" * 50)
    print("\nHow to read the heatmap:")
    print("  🔴 Red/orange areas — the model focused most here")
    print("  🟡 Yellow areas     — moderately important")
    print("  🔵 Blue areas       — mostly ignored by the model")
    print()
    if pred_label == "PNEUMONIA":
        print("  Tip: For pneumonia cases, look for red areas in the lung fields.")
        print("  Pneumonia typically causes opacities in the lower lobes.")
    else:
        print("  Tip: For normal cases, the model should focus on clear lung fields.")
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gradcam.py <path_to_xray_image>")
        print("Example: python gradcam.py data/test/PNEUMONIA/person1_virus_1.jpeg")
        sys.exit(1)

    run_gradcam(image_path=sys.argv[1])
