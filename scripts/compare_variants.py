"""Cross-variant Grad-CAM comparison: same test images, multiple checkpoints, side by side.

Examples (PowerShell):

    # architectural ablation
    python scripts/compare_variants.py --checkpoints `
        runs/vanilla_scratch/checkpoints/best.pth `
        runs/no_se_scratch/checkpoints/best.pth `
        runs/small_kernel_scratch/checkpoints/best.pth --num-images 6 --only-all-correct

    # transfer-learning axis
    python scripts/compare_variants.py --checkpoints `
        runs/vanilla_scratch/checkpoints/best.pth `
        runs/vanilla_finetune/checkpoints/best.pth --num-images 6
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import CIFAR10_CLASSES, build_loaders
from src.explain import build_comparison, render_comparison_grid, select_shared_indices
from src.models import VARIANTS, build_mobilenetv3_small


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same test images through multiple checkpoints and compare Grad-CAMs side by side."
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, help="Paths to Phase 2 .pth checkpoints, one per model."
    )
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--target", choices=["true", "pred"], default="true")
    parser.add_argument(
        "--only-all-correct",
        action="store_true",
        help="Restrict candidate images to those every model classifies correctly.",
    )
    parser.add_argument("--output-dir", default="runs/compare")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--no-download", action="store_true", help="Use a synthetic random test set instead of CIFAR-10."
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class SyntheticTestSet(torch.utils.data.Dataset):
    """Un-downloaded fallback: random normalized-looking images with random labels."""

    def __init__(self, n: int = 64, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, 224, 224, generator=g) * 0.25
        self.labels = torch.randint(0, num_classes, (n,), generator=g).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is not None:
        variant = config["model"]["variant"]
        num_classes = config["model"]["num_classes"]
    else:
        variant = VARIANTS[0]
        num_classes = len(CIFAR10_CLASSES)

    model = build_mobilenetv3_small(variant=variant, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    experiment_dir = checkpoint_path.resolve().parent.parent
    display_name = experiment_dir.name
    return model, display_name


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_and_names = []
    for ckpt in args.checkpoints:
        model, name = load_model(Path(ckpt), device)
        models_and_names.append((model, name))

    if args.no_download:
        test_set = SyntheticTestSet(n=max(64, args.num_images * 4), num_classes=len(CIFAR10_CLASSES), seed=args.seed)
    else:
        _, test_loader = build_loaders(root=args.data_root, num_workers=0, download=True)
        test_set = test_loader.dataset

    indices = select_shared_indices(
        test_set,
        models_and_names,
        device,
        num_images=args.num_images,
        seed=args.seed,
        only_all_correct=args.only_all_correct,
    )
    print(f"Selected {len(indices)} shared test images: {indices}")

    result = build_comparison(models_and_names, test_set, indices, device, target=args.target)

    grid_name = "_vs_".join(name for _, name in models_and_names) + f"_{args.target}.png"
    out_path = output_dir / grid_name
    render_comparison_grid(result, CIFAR10_CLASSES, out_path)

    print(f"Saved comparison grid to {out_path.resolve()}")
    print("Per-model accuracy on the shared image set:")
    for _, name in models_and_names:
        correct = sum(
            1
            for item in result
            for pm in item["per_model"]
            if pm["name"] == name and pm["pred_label"] == item["true_label"]
        )
        total = len(result)
        acc = correct / total if total else float("nan")
        print(f"  {name}: {correct}/{total} ({acc:.1%})")


if __name__ == "__main__":
    main()
