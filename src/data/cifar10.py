"""CIFAR-10 data pipeline: transforms, loaders, and denormalization for overlays.

Images are upsampled from CIFAR-10's native 32x32 to 224x224 (INPUT_SIZE in
src.utils) so the MobileNetV3-Small variants -- built for ImageNet-scale
inputs -- receive a resolution consistent with their expected receptive
field, and so Grad-CAM/robustness code shares one canonical input size.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
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


def _stratified_val_split(
    targets: List[int], val_size: int, split_seed: int
) -> Tuple[List[int], List[int]]:
    """Carve a class-stratified validation index set out of `targets`.

    Uses a LOCAL `numpy.random.default_rng(split_seed)` -- never the global
    NumPy RNG that `src.utils.set_seed` seeds -- so the split is completely
    independent of the training seed: every training run sees the identical
    validation set regardless of `--seed` (only model init and loader
    shuffling vary across seeds). Returns (train_indices, val_indices), both
    sorted and disjoint.
    """
    n_total = len(targets)
    classes = sorted({int(t) for t in targets})
    num_classes = len(classes)
    per_class = val_size // num_classes
    remainder = val_size - per_class * num_classes

    class_to_indices: Dict[int, List[int]] = {c: [] for c in classes}
    for i, t in enumerate(targets):
        class_to_indices[int(t)].append(i)

    rng = np.random.default_rng(split_seed)
    val_indices: List[int] = []
    for j, c in enumerate(classes):
        pool = class_to_indices[c]
        take = per_class + (1 if j < remainder else 0)
        chosen = rng.choice(len(pool), size=take, replace=False)
        val_indices.extend(pool[k] for k in chosen)

    val_set = set(val_indices)
    train_indices = [i for i in range(n_total) if i not in val_set]
    return sorted(train_indices), sorted(val_indices)


def build_loaders(
    root: str,
    train_batch_size: int = 128,
    test_batch_size: int = 32,
    num_workers: int = 4,
    download: bool = True,
    worker_init_fn: Optional[Callable] = None,
    val_size: int = 5000,
    split_seed: int = 1234,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build CIFAR-10 train/val/test DataLoaders.

    `val_size` examples are held out of the 50k-image training set as a
    stratified validation split (see `_stratified_val_split`), leaving the
    remainder for training; the original 10k-image test set is untouched and
    used only for a final, post-selection accuracy report -- never for
    checkpoint selection. The split is keyed on `split_seed`, not the
    training seed, so every training run (any `--seed`) sees the identical
    train/val partition.

    Downloads to `root` on first use if `download=True`. `worker_init_fn`
    should be `src.utils.seed.seed_worker` when reproducibility across
    dataloader workers is required (see scripts/train.py).
    """
    train_base = CIFAR10(
        root=root, train=True, download=download, transform=build_train_transform()
    )
    val_base = CIFAR10(
        root=root, train=True, download=False, transform=build_eval_transform()
    )
    test_set = CIFAR10(
        root=root, train=False, download=download, transform=build_eval_transform()
    )

    train_indices, val_indices = _stratified_val_split(train_base.targets, val_size, split_seed)
    train_subset = Subset(train_base, train_indices)
    val_subset = Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=test_batch_size,
        shuffle=False,
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
    return train_loader, val_loader, test_loader
