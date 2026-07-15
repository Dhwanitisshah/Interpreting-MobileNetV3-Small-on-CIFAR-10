import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import build_mobilenetv3_small
from src.train import evaluate, save_eval_artifacts, train_model
from src.utils import DotDict, set_seed

NUM_CLASSES = 10
NUM_SAMPLES = 64


def build_synthetic_loaders() -> tuple:
    images = torch.randn(NUM_SAMPLES, 3, 224, 224)
    labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(images, labels)

    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    return train_loader, val_loader, dataset


def build_cfg() -> DotDict:
    return DotDict(
        {
            "experiment": "smoke_test",
            "seed": 42,
            "model": {"variant": "vanilla", "pretrained": False, "num_classes": NUM_CLASSES},
            "train": {"epochs": 2, "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
        }
    )


def main() -> bool:
    set_seed(42)
    results = []
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "run"

        train_loader, val_loader, dataset = build_synthetic_loaders()
        cfg = build_cfg()
        model = build_mobilenetv3_small(
            variant=cfg.model.variant,
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
        )

        # 1: train_model runs and produces finite losses
        history = train_model(
            model,
            train_loader,
            val_loader,
            cfg,
            device,
            str(output_dir),
            limit_train_batches=2,
            limit_val_batches=2,
        )
        losses_finite = all(
            math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]) for h in history
        )
        results.append(("train_model completes with 2 epochs", len(history) == 2))
        results.append(("all logged losses are finite", losses_finite))

        # 2: checkpoints written and reloadable
        last_ckpt_path = output_dir / "checkpoints" / "last.pth"
        best_ckpt_path = output_dir / "checkpoints" / "best.pth"
        results.append(("checkpoints/last.pth exists", last_ckpt_path.exists()))
        results.append(("checkpoints/best.pth exists", best_ckpt_path.exists()))

        last_ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        reload_ok = "model_state" in last_ckpt and "model_state" in best_ckpt
        results.append(("checkpoints reload with model_state key", reload_ok))

        # 3: evaluate on synthetic val loader
        eval_result = evaluate(model, val_loader, device, num_classes=NUM_CLASSES)
        results.append(
            ("per_class_acc has length 10", len(eval_result["per_class_acc"]) == NUM_CLASSES)
        )
        results.append(
            (
                "confusion matrix is 10x10",
                len(eval_result["confusion"]) == NUM_CLASSES
                and all(len(row) == NUM_CLASSES for row in eval_result["confusion"]),
            )
        )
        results.append(
            ("len(records) == dataset size", len(eval_result["records"]) == len(dataset))
        )

        # 4: save_eval_artifacts partitions the set
        artifacts_dir = Path(tmpdir) / "artifacts"
        save_eval_artifacts(eval_result, str(artifacts_dir))

        import json

        with open(artifacts_dir / "eval" / "correct_indices.json") as f:
            correct_indices = json.load(f)
        with open(artifacts_dir / "eval" / "incorrect_indices.json") as f:
            incorrect_indices = json.load(f)

        partition_ok = len(correct_indices) + len(incorrect_indices) == len(dataset)
        results.append(("correct + incorrect indices partition the dataset", partition_ok))

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
