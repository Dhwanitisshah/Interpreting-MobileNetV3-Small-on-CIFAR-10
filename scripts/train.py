import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.data import build_loaders
from src.models import build_mobilenetv3_small
from src.train import run_final_test_evaluation, save_eval_artifacts, train_model
from src.utils import SyntheticTestSet, load_config, resolve_device, set_seed
from src.utils.seed import seed_worker

# Fixed seed for the synthetic --no-download val/test sets, mirroring
# build_loaders' split_seed default: independent of the training seed so it
# never varies across --seed values (train.py's --no-download path is for
# smoke testing only; see scripts/smoke_test_campaign.py).
NO_DOWNLOAD_SPLIT_SEED = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a MobileNetV3 variant.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device to train on."
    )
    parser.add_argument(
        "--output-root", default="runs", help="Root directory for run outputs."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override cfg.seed (output goes to runs/<experiment>/seed<N>/)."
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Override cfg.data.num_workers."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override cfg.train.epochs.")
    parser.add_argument(
        "--limit-train-batches", type=int, default=None, help="Limit train batches per epoch."
    )
    parser.add_argument(
        "--limit-val-batches", type=int, default=None, help="Limit val batches per epoch."
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use a small synthetic random dataset instead of downloading CIFAR-10 (smoke testing only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    resolved_seed = args.seed if args.seed is not None else cfg.seed
    cfg.seed = resolved_seed
    set_seed(resolved_seed)

    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    device = resolve_device(args.device)
    output_dir = Path(args.output_root) / cfg.experiment / f"seed{resolved_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Seed: {resolved_seed}")
    print(f"Output dir: {output_dir}")

    if args.no_download:
        train_set = SyntheticTestSet(n=64, num_classes=cfg.model.num_classes, seed=resolved_seed)
        val_set = SyntheticTestSet(n=32, num_classes=cfg.model.num_classes, seed=NO_DOWNLOAD_SPLIT_SEED)
        test_set = SyntheticTestSet(n=64, num_classes=cfg.model.num_classes, seed=NO_DOWNLOAD_SPLIT_SEED + 1)
        train_loader = DataLoader(train_set, batch_size=cfg.data.train_batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=cfg.data.test_batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=cfg.data.test_batch_size, shuffle=False, num_workers=0)
    else:
        train_loader, val_loader, test_loader = build_loaders(
            root=cfg.data.root,
            train_batch_size=cfg.data.train_batch_size,
            test_batch_size=cfg.data.test_batch_size,
            num_workers=cfg.data.num_workers,
            download=True,
            worker_init_fn=seed_worker,
        )

    model = build_mobilenetv3_small(
        variant=cfg.model.variant,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )

    history = train_model(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        str(output_dir),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    best_val_acc = max((h["val_acc"] for h in history), default=0.0)

    eval_result = run_final_test_evaluation(
        model, test_loader, device, str(output_dir), num_classes=cfg.model.num_classes
    )
    save_eval_artifacts(eval_result, str(output_dir))

    print("\n=== SUMMARY ===")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test acc: {eval_result['overall_acc']:.4f}")
    print(f"Per-class acc: {eval_result['per_class_acc']}")


if __name__ == "__main__":
    main()
