import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch

from src.data import CIFAR10_CLASSES, IMAGENET_MEAN, IMAGENET_STD, build_loaders, vis_transform
from src.explain import GradCAM, cascading_randomization, overlay_cam
from src.models import VARIANTS, build_mobilenetv3_small


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adebayo et al. (2018) cascading parameter-randomization sanity check for Grad-CAM."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a Phase 2 .pth checkpoint.")
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--output-dir", default=None, help="Default: runs/sanity/<experiment>.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--no-download", action="store_true", help="Use synthetic random images instead of CIFAR-10."
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
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
    return model


def normalize_for_model(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, -1, 1, 1)
    return (images - mean) / std


def collect_images(args: argparse.Namespace, experiment_dir: Path) -> tuple:
    n = args.num_images

    if args.no_download:
        images = torch.rand(n, 3, 224, 224)
        labels = [None] * n
        return images, labels

    _, test_loader = build_loaders(
        root=args.data_root,
        test_batch_size=n,
        num_workers=0,
        download=True,
    )
    vis_set = test_loader.dataset.__class__(
        root=args.data_root, train=False, download=False, transform=vis_transform()
    )

    correct_indices_path = experiment_dir / "eval" / "correct_indices.json"
    if correct_indices_path.exists():
        with open(correct_indices_path) as f:
            indices = json.load(f)[:n]
        print(f"Using {len(indices)} correctly-classified test images from {correct_indices_path}")
    else:
        indices = list(range(n))
        print(f"No eval artifacts found at {correct_indices_path}; using the first {n} test images.")

    images = torch.stack([vis_set[i][0] for i in indices])
    labels = [vis_set[i][1] for i in indices]
    return images, labels


def make_grid_figure(
    vis_images: torch.Tensor,
    reference_cams: torch.Tensor,
    steps: list,
    labels: list,
    output_path: Path,
) -> None:
    n_images = vis_images.shape[0]
    n_cols = 1 + len(steps)

    fig, axes = plt.subplots(n_images, n_cols, figsize=(1.6 * n_cols, 1.6 * n_images), squeeze=False)

    for row in range(n_images):
        image = vis_images[row].permute(1, 2, 0).numpy()

        overlay = overlay_cam(image, reference_cams[row].numpy())
        axes[row][0].imshow(overlay)
        axes[row][0].axis("off")
        if row == 0:
            axes[row][0].set_title("Original CAM", fontsize=9)
        if labels[row] is not None:
            axes[row][0].set_ylabel(CIFAR10_CLASSES[labels[row]], fontsize=8)

        for col, step in enumerate(steps, start=1):
            overlay = overlay_cam(image, step["cams"][row].numpy())
            axes[row][col].imshow(overlay)
            axes[row][col].axis("off")
            if row == 0:
                title = step["step_name"] if col < n_cols - 1 else "Fully Random"
                axes[row][col].set_title(title, fontsize=8, rotation=45, ha="left")

    fig.suptitle("Cascading Parameter Randomization (top-down)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def make_decay_plot(steps: list, output_path: Path) -> None:
    x = list(range(len(steps)))
    x_labels = [s["step_name"] for s in steps]
    spearman_vals = [s["mean_spearman"] for s in steps]
    ssim_vals = [s["mean_ssim"] for s in steps]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(steps)), 4))
    ax.plot(x, spearman_vals, marker="o", label="mean Spearman")
    ax.plot(x, ssim_vals, marker="s", label="mean SSIM")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Cascade step (top-down)")
    ax.set_ylabel("Similarity to original CAM")
    ax.set_title("Grad-CAM Sanity Check: Cascading Randomization Decay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    experiment_dir = checkpoint_path.resolve().parent.parent
    experiment_name = experiment_dir.name

    output_dir = Path(args.output_dir) if args.output_dir else Path("runs/sanity") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path, device)
    vis_images, labels = collect_images(args, experiment_dir)
    model_input = normalize_for_model(vis_images).to(device)

    steps, reference_cams = cascading_randomization(model, model_input, seed=args.seed)

    grid_path = output_dir / "sanity_grid.png"
    make_grid_figure(vis_images, reference_cams, steps, labels, grid_path)

    decay_path = output_dir / "decay_plot.png"
    make_decay_plot(steps, decay_path)

    metrics = [
        {"step_name": s["step_name"], "mean_spearman": s["mean_spearman"], "mean_ssim": s["mean_ssim"]}
        for s in steps
    ]
    metrics_path = output_dir / "sanity_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved grid figure to {grid_path.resolve()}")
    print(f"Saved decay plot to {decay_path.resolve()}")
    print(f"Saved metrics to {metrics_path.resolve()}")

    final_step = steps[-1]
    print(
        f"Final step ({final_step['step_name']}, fully random): "
        f"mean_spearman={final_step['mean_spearman']:.4f}, mean_ssim={final_step['mean_ssim']:.4f}"
    )


if __name__ == "__main__":
    main()
