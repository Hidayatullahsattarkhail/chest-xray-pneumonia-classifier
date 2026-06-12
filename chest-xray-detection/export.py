"""
export.py — Export the trained model to TorchScript for production deployment.

TorchScript converts the PyTorch model into a portable, optimized format
that can run without Python — useful for deploying with FastAPI, C++ apps,
or mobile devices.

Usage:
    python export.py

Output:
    densenet121_pneumonia_prod.pt — TorchScript model ready for deployment

Deploy with FastAPI example:
    import torch
    model = torch.jit.load("densenet121_pneumonia_prod.pt")
    model.eval()
    # Pass a (1, 3, 224, 224) tensor to get predictions
"""

import torch
from model import load_model


# ── Configuration ─────────────────────────────────────────────────────────────

CHECKPOINT   = "best_model.pth"
EXPORT_PATH  = "densenet121_pneumonia_prod.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export():
    """
    Load the trained model and save it as a TorchScript file.

    TorchScript tracing works by running a dummy input through the model
    and recording every operation — the result is a self-contained graph
    that doesn't need the original Python code to run.
    """
    if not __import__("os").path.isfile(CHECKPOINT):
        print(f"ERROR: '{CHECKPOINT}' not found.")
        print("Please run train.py first to train and save the model.")
        return

    print(f"Loading model from: {CHECKPOINT}")
    model = load_model(CHECKPOINT, device)
    model.eval()

    # Create a dummy input with the same shape the model expects
    # Shape: (batch=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print("Tracing model with TorchScript...")
    traced_model = torch.jit.trace(model, dummy_input)

    traced_model.save(EXPORT_PATH)

    print()
    print("=" * 50)
    print(f"  TorchScript model saved to: {EXPORT_PATH}")
    print("=" * 50)
    print()
    print("To load and use this model anywhere (no Python class needed):")
    print()
    print("  import torch")
    print(f"  model = torch.jit.load('{EXPORT_PATH}')")
    print("  model.eval()")
    print("  output = model(input_tensor)  # shape: (1, 3, 224, 224)")
    print()
    print("This file can be deployed with:")
    print("  • FastAPI + Python")
    print("  • LibTorch (C++ runtime)")
    print("  • TorchServe")


if __name__ == "__main__":
    export()
