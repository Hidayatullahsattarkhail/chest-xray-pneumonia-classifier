"""
model.py — Defines the chest X-ray classification model.
Uses a pretrained ResNet18 and replaces the final layer for binary
classification: NORMAL vs PNEUMONIA.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(use_pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet18 model fine-tuned for binary classification.

    Args:
        use_pretrained: If True, loads ImageNet weights (recommended).

    Returns:
        A PyTorch model ready for training or inference.
    """
    # Load pretrained ResNet18
    weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze all layers so we only train the final classifier
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer
    # ResNet18's fc layer outputs 512 features → we map to 2 classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),  # 2 classes: NORMAL, PNEUMONIA
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
