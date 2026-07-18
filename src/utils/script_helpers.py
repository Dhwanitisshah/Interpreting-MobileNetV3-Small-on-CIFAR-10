"""Shared helpers for `scripts/*.py` entry points.

These were originally copy-pasted (with minor drift) across compare_variants.py,
faithfulness_eval.py, robustness_eval.py, gradcam_demo.py, sanity_check.py, and
concentration_diagnostic.py: device resolution, checkpoint loading with the
Phase 6.1 provenance check, a synthetic (no-download) dataset fallback,
stratified index sampling, and the shared plotting palette. Consolidated here
so behavior stays identical across scripts and only needs to be fixed in one
place.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.data import CIFAR10_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from src.models import VARIANTS, build_mobilenetv3_small

# Input resolution every model variant expects (CIFAR-10 images are upsampled
# to this size in src.data.cifar10; see build_train_transform/build_eval_transform).
INPUT_SIZE = 224

# Relative tolerance, in val_acc, before a checkpoint is considered to have
# drifted from the metrics.json recorded by its own training run.
CHECKPOINT_VAL_ACC_TOLERANCE = 0.02

# Shared categorical palette so the same model always gets the same color
# across every figure (faithfulness, robustness, comparison plots).
CATEGORICAL_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"

# Publication-quality raster resolution for every saved figure.
FIGURE_DPI = 300


def set_publication_style() -> None:
    """Apply consistent, publication-quality matplotlib rcParams: fonts, sizes,
    and save resolution. Call once near the start of a script's main(), before
    any figure is created, so every plot in the paper shares one visual style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "savefig.dpi": FIGURE_DPI,
            "figure.dpi": 100,  # on-screen only; file output is controlled by savefig.dpi
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
        }
    )
MUTED_TEXT = "#898781"


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the `--device` CLI choice ("auto", "cpu", "cuda") to a torch.device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def normalize_for_model(images: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std normalization to a batch of [0, 1] visualization images."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, -1, 1, 1)
    return (images - mean) / std


class SyntheticTestSet(torch.utils.data.Dataset):
    """No-download fallback: random normalized-looking images with random labels.

    Used by every `scripts/*.py --no-download` path so smoke-testing a full
    pipeline doesn't require the CIFAR-10 archive.
    """

    def __init__(self, n: int = 512, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, INPUT_SIZE, INPUT_SIZE, generator=g) * 0.25
        self.labels = torch.randint(0, num_classes, (n,), generator=g).tolist()
        self.targets = self.labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx]


def select_indices(dataset, num_images: int, seed: int, stratified: bool) -> List[int]:
    """One shared, seeded set of dataset indices, optionally stratified by class.

    Used to build a paired design: the same images are evaluated across every
    model/checkpoint being compared.
    """
    n_total = len(dataset)
    num_images = min(num_images, n_total)
    rng = random.Random(seed)

    if not stratified:
        return sorted(rng.sample(range(n_total), num_images))

    targets = dataset.targets if hasattr(dataset, "targets") else [dataset[i][1] for i in range(n_total)]
    num_classes = len(CIFAR10_CLASSES)
    per_class = num_images // num_classes
    remainder = num_images - per_class * num_classes

    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, t in enumerate(targets):
        class_to_indices[int(t)].append(i)

    indices: List[int] = []
    for c in range(num_classes):
        pool = class_to_indices[c]
        take = min(per_class + (1 if c < remainder else 0), len(pool))
        indices.extend(rng.sample(pool, take))

    return sorted(indices)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    check_provenance: bool = False,
) -> Tuple[torch.nn.Module, str]:
    """Rebuild a model from a `scripts/train.py` checkpoint.

    Loading is strict. If `check_provenance` is set, the checkpoint's own
    recorded val_acc is cross-checked against its experiment's metrics.json
    (the Phase 6.1 check) so a stale/mismatched checkpoint on disk fails
    loudly here instead of silently producing misleading downstream numbers.

    Returns (model, display_name), where display_name is the experiment
    directory name (e.g. "vanilla_scratch") derived from the checkpoint path.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is not None:
        variant = config["model"]["variant"]
        num_classes = config["model"]["num_classes"]
    else:
        variant = VARIANTS[0]
        num_classes = len(CIFAR10_CLASSES)

    model = build_mobilenetv3_small(variant=variant, num_classes=num_classes, pretrained=False)
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to strictly load checkpoint '{checkpoint_path}' (variant='{variant}'): {e}"
        ) from e
    model.to(device)
    model.eval()

    experiment_dir = checkpoint_path.resolve().parent.parent
    display_name = experiment_dir.name

    if check_provenance:
        metrics_path = experiment_dir / "metrics.json"
        ckpt_val_acc = checkpoint.get("val_acc")
        if metrics_path.exists() and ckpt_val_acc is not None:
            with open(metrics_path) as f:
                summary = json.load(f)["summary"]
            if abs(ckpt_val_acc - summary["best_val_acc"]) > CHECKPOINT_VAL_ACC_TOLERANCE:
                raise RuntimeError(
                    f"Checkpoint provenance mismatch for '{display_name}': {checkpoint_path} stores "
                    f"val_acc={ckpt_val_acc:.4f} (epoch {checkpoint.get('epoch')}) but {metrics_path} "
                    f"records best_val_acc={summary['best_val_acc']:.4f} (best_epoch {summary['best_epoch']}). "
                    f"The checkpoint on disk does not match its own training run's metrics.json -- "
                    f"regenerate it, e.g. `python scripts/train.py --config configs/{display_name}.yaml`, "
                    f"before trusting downstream numbers."
                )

    return model, display_name
