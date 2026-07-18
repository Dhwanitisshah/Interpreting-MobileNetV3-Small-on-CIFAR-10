import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_loaders
from src.models import build_mobilenetv3_small
from src.train import evaluate, save_eval_artifacts, train_model
from src.utils import load_config, resolve_device, set_seed
from src.utils.seed import seed_worker


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
        "--num-workers", type=int, default=None, help="Override cfg.data.num_workers."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override cfg.train.epochs.")
    parser.add_argument(
        "--limit-train-batches", type=int, default=None, help="Limit train batches per epoch."
    )
    parser.add_argument(
        "--limit-val-batches", type=int, default=None, help="Limit val batches per epoch."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    device = resolve_device(args.device)
    output_dir = Path(args.output_root) / cfg.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_loaders(
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
        test_loader,
        cfg,
        device,
        str(output_dir),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    best_val_acc = max((h["val_acc"] for h in history), default=0.0)

    eval_result = evaluate(model, test_loader, device, num_classes=cfg.model.num_classes)
    save_eval_artifacts(eval_result, str(output_dir))

    print("\n=== SUMMARY ===")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test acc: {eval_result['overall_acc']:.4f}")
    print(f"Per-class acc: {eval_result['per_class_acc']}")


if __name__ == "__main__":
    main()
