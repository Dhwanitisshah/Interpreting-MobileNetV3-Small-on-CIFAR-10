"""CIFAR-10 data pipeline: transforms, loaders, and denormalization for overlays.

Images are upsampled from CIFAR-10's native 32x32 to 224x224 (INPUT_SIZE in
src.utils) so the MobileNetV3-Small variants -- built for ImageNet-scale
inputs -- receive a resolution consistent with their expected receptive
field, and so Grad-CAM/robustness code shares one canonical input size.
"""

from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform() -> transforms.Compose:
    """Train-time augmentation: crop/flip at native 32x32, then upsample to
    224x224 and apply ImageNet normalization (see module docstring)."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Deterministic eval-time transform: upsample to 224x224 and normalize, no augmentation."""
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def vis_transform() -> transforms.Compose:
    """Resize-only transform producing un-normalized [0, 1] tensors for overlays."""
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization and clamp to [0, 1]. Accepts (C,H,W) or (N,C,H,W)."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device)
    if x.dim() == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif x.dim() == 3:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    else:
        raise ValueError(f"Expected a (C,H,W) or (N,C,H,W) tensor, got shape {tuple(x.shape)}")
    return (x * std + mean).clamp(0.0, 1.0)


def build_loaders(
    root: str,
    train_batch_size: int = 128,
    test_batch_size: int = 32,
    num_workers: int = 4,
    download: bool = True,
    worker_init_fn: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build CIFAR-10 train/test DataLoaders with the standard train/eval transforms.

    Downloads to `root` on first use if `download=True`. `worker_init_fn` should
    be `src.utils.seed.seed_worker` when reproducibility across dataloader
    workers is required (see scripts/train.py).
    """
    train_set = CIFAR10(
        root=root, train=True, download=download, transform=build_train_transform()
    )
    test_set = CIFAR10(
        root=root, train=False, download=download, transform=build_eval_transform()
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, test_loader
