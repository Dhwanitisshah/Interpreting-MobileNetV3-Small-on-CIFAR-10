"""Smoke test for the Phase 8.0 train/val/test split and val-based checkpoint
selection. Runs with no CIFAR-10 download: the split logic is exercised
through `build_loaders` against a CIFAR10-shaped fake dataset, and checkpoint
selection is exercised with synthetic tensors, exactly like the other
`smoke_test_*.py` scripts.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms as T

import src.data.cifar10 as cifar10_module
from src.data import build_loaders
from src.models import build_mobilenetv3_small
from src.train import run_final_test_evaluation, train_model
from src.utils import DotDict, set_seed

NUM_CLASSES = 10


class FakeCIFAR10(Dataset):
    """CIFAR10-shaped stand-in (50000 train / 10000 test images, real class
    count) so `build_loaders`'s split/transform logic can be exercised without
    downloading the real archive."""

    def __init__(self, root, train=True, download=False, transform=None):
        self.transform = transform
        n = 50000 if train else 10000
        rng = np.random.default_rng(0 if train else 1)
        self.targets = rng.integers(0, NUM_CLASSES, size=n).tolist()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        target = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def has_random_crop(compose: T.Compose) -> bool:
    return any(isinstance(t, T.RandomCrop) for t in compose.transforms)


def build_synthetic_loader(n: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, 224, 224, generator=g)
    labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)


def build_train_cfg() -> DotDict:
    return DotDict(
        {
            "experiment": "smoke_test_split",
            "seed": 42,
            "model": {"variant": "vanilla", "pretrained": False, "num_classes": NUM_CLASSES},
            "train": {"epochs": 1, "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
        }
    )


def main() -> bool:
    set_seed(42)
    results = []
    device = torch.device("cpu")

    # --- Part A: build_loaders split, determinism, seed-independence, transforms ---
    original_cifar10 = cifar10_module.CIFAR10
    cifar10_module.CIFAR10 = FakeCIFAR10
    try:
        train_loader, val_loader, test_loader = build_loaders(
            root="unused", num_workers=0, download=False, val_size=5000, split_seed=1234
        )

        results.append(("train split size == 45000", len(train_loader.dataset) == 45000))
        results.append(("val split size == 5000", len(val_loader.dataset) == 5000))
        results.append(("test split size == 10000", len(test_loader.dataset) == 10000))

        train_idx = set(train_loader.dataset.indices)
        val_idx = set(val_loader.dataset.indices)
        results.append(("train/val indices are disjoint", train_idx.isdisjoint(val_idx)))
        results.append(
            ("train/val indices partition the 50k training set", train_idx | val_idx == set(range(50000)))
        )

        def val_indices(split_seed: int) -> list:
            _, v_loader, _ = build_loaders(
                root="unused", num_workers=0, download=False, val_size=5000, split_seed=split_seed
            )
            return sorted(v_loader.dataset.indices)

        results.append(
            ("determinism: two calls with the same split_seed give identical val indices",
             val_indices(1234) == val_indices(1234))
        )

        set_seed(1)
        val_idx_seed1 = val_indices(1234)
        set_seed(2)
        val_idx_seed2 = val_indices(1234)
        results.append(
            ("seed-independence: global set_seed(1) vs set_seed(2) give identical val indices",
             val_idx_seed1 == val_idx_seed2)
        )

        results.append(
            ("train loader uses the augmenting (RandomCrop) transform",
             has_random_crop(train_loader.dataset.dataset.transform))
        )
        results.append(
            ("val loader uses the non-augmenting transform",
             not has_random_crop(val_loader.dataset.dataset.transform))
        )
        results.append(
            ("test loader uses the non-augmenting transform",
             not has_random_crop(test_loader.dataset.transform))
        )
    finally:
        cifar10_module.CIFAR10 = original_cifar10

    # --- Part B: a 1-epoch run selects best.pth on val, not test ---
    set_seed(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "run"

        train_loader = build_synthetic_loader(16, seed=10)
        val_loader = build_synthetic_loader(16, seed=20)
        test_loader = build_synthetic_loader(24, seed=30)  # deliberately different size than val

        cfg = build_train_cfg()
        model = build_mobilenetv3_small(
            variant=cfg.model.variant, num_classes=cfg.model.num_classes, pretrained=cfg.model.pretrained
        )

        history = train_model(
            model, train_loader, val_loader, cfg, device, str(output_dir),
            limit_train_batches=2, limit_val_batches=2,
        )
        best_val_acc = max(h["val_acc"] for h in history)

        with open(output_dir / "metrics.json") as f:
            metrics_before = json.load(f)
        results.append(
            ("metrics.json summary's best_val_acc matches the val-loader history",
             math.isclose(metrics_before["summary"]["best_val_acc"], best_val_acc))
        )
        results.append(
            ("metrics.json has no test_acc before the final test evaluation runs",
             "test_acc" not in metrics_before["summary"])
        )

        eval_result = run_final_test_evaluation(
            model, test_loader, device, str(output_dir), num_classes=NUM_CLASSES
        )
        results.append(
            ("final test evaluation covers exactly the 24-example test set, not the 16-example val set",
             len(eval_result["records"]) == 24)
        )

        with open(output_dir / "metrics.json") as f:
            metrics_after = json.load(f)
        results.append(
            ("metrics.json summary now records both best_val_acc and test_acc",
             "best_val_acc" in metrics_after["summary"] and "test_acc" in metrics_after["summary"])
        )
        results.append(
            ("best_val_acc is unchanged by the test-set evaluation",
             metrics_after["summary"]["best_val_acc"] == metrics_before["summary"]["best_val_acc"])
        )
        results.append(
            ("recorded test_acc matches the test-set evaluation's own accuracy",
             math.isclose(metrics_after["summary"]["test_acc"], eval_result["overall_acc"]))
        )

        best_ckpt = torch.load(
            output_dir / "checkpoints" / "best.pth", map_location="cpu", weights_only=False
        )
        results.append(
            ("best.pth's recorded val_acc matches metrics.json's best_val_acc (selection provenance)",
             math.isclose(best_ckpt["val_acc"], metrics_after["summary"]["best_val_acc"]))
        )

    print("\n=== SMOKE TEST RESULTS ===")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"[{status}] {name}")

    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
