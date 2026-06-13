"""
dataset.py — Loads and preprocesses the chest X-ray dataset.

Dataset structure expected (from Kaggle chest-xray-pneumonia):
    data/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from config import IMAGE_SIZE, CLASS_NAMES


def get_transforms(split: str) -> transforms.Compose:
    """
    Return image transforms for a given dataset split.

    Training uses data augmentation to help the model generalise.
    Validation/test use only resizing and normalization.

    Args:
        split: One of "train", "val", or "test".

    Returns:
        A torchvision Compose transform pipeline.
    """
    # ImageNet mean and std — required when using pretrained ResNet weights
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict:
    """
    Create DataLoader objects for train, val, and test splits.

    Args:
        data_dir:    Root folder containing train/, val/, test/ subfolders.
        batch_size:  Number of images per batch.
        num_workers: Parallel data loading workers (set to 0 on Windows if issues).

    Returns:
        A dict with keys "train", "val", "test" mapping to DataLoader objects.
        Also returns dataset sizes and class names for convenience.
    """
    splits = ["train", "val", "test"]
    dataloaders = {}
    dataset_sizes = {}

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: '{split_dir}' not found — skipping '{split}' split.")
            continue

        dataset = datasets.ImageFolder(
            root=split_dir,
            transform=get_transforms(split),
        )

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        dataset_sizes[split] = len(dataset)

    print("Dataset sizes:", dataset_sizes)
    print("Classes:", CLASS_NAMES)
    return dataloaders, dataset_sizes
