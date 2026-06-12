"""
model.py — Defines the chest X-ray classification model.

Uses a pretrained DenseNet121 with a custom classifier head for binary
classification: NORMAL vs PNEUMONIA.

DenseNet121 is commonly used in medical imaging research and often
outperforms ResNet18 on chest X-ray tasks.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(use_pretrained: bool = True) -> nn.Module:
    """
    Build a DenseNet121 model fine-tuned for binary classification.

    Args:
        use_pretrained: If True, loads ImageNet weights (recommended).

    Returns:
        A PyTorch model ready for training or inference.
    """
    # Load pretrained DenseNet121
    weights = models.DenseNet121_Weights.DEFAULT if use_pretrained else None
    model = models.densenet121(weights=weights)

    # Freeze all layers so we only train the final classifier
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head
    # DenseNet121's classifier outputs 1024 features → we map to 2 classes
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2),  # 2 classes: NORMAL, PNEUMONIA
    )

    return model


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load a saved model checkpoint from disk.

    Args:
        checkpoint_path: Path to the .pth file saved during training.
        device: torch.device("cpu") or torch.device("cuda").

    Returns:
        The model loaded with saved weights, set to eval mode.
    """
    model = build_model(use_pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    return model


def get_gradcam_target_layer(model: nn.Module):
    """
    Return the target layer for Grad-CAM visualisation.

    For DenseNet121, the last feature block contains the most
    semantically meaningful spatial information.

    Returns:
        The target layer (torch.nn.Module).
    """
    return model.features[-1]
