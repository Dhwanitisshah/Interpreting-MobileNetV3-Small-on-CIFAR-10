import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch

from src.data import CIFAR10_CLASSES, build_loaders, denormalize, vis_transform
from src.explain import GradCAM, overlay_cam
from src.models import VARIANTS, build_mobilenetv3_small


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save Grad-CAM overlay panels for a model.")
    parser.add_argument("--checkpoint", default=None, help="Path to a Phase 2 .pth checkpoint.")
    parser.add_argument(
        "--random", action="store_true", help="Build an untrained model instead of loading a checkpoint."
    )
    parser.add_argument(
        "--variant", choices=VARIANTS, default="vanilla", help="Fallback variant if checkpoint has no config."
    )
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--output-dir", default="runs/gradcam_demo")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--no-download", action="store_true", help="Use synthetic random images instead of CIFAR-10."
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("config")
        if config is not None:
            variant = config["model"]["variant"]
            num_classes = config["model"]["num_classes"]
        else:
            variant = args.variant
            num_classes = len(CIFAR10_CLASSES)
        model = build_mobilenetv3_small(variant=variant, num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        return model

    return build_mobilenetv3_small(variant=args.variant, num_classes=len(CIFAR10_CLASSES), pretrained=False)


def collect_images(args: argparse.Namespace) -> tuple:
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
    images = torch.stack([vis_set[i][0] for i in range(n)])
    labels = [vis_set[i][1] for i in range(n)]
    return images, labels


def normalize_for_model(images: torch.Tensor) -> torch.Tensor:
    from src.data import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, -1, 1, 1)
    return (images - mean) / std


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args)
    model.eval()

    vis_images, labels = collect_images(args)
    model_input = normalize_for_model(vis_images)

    with GradCAM(model) as gradcam:
        cams, pred_classes = gradcam(model_input)

    for i in range(vis_images.shape[0]):
        image = vis_images[i].permute(1, 2, 0).numpy()
        cam = cams[i].numpy()
        overlay = overlay_cam(image, cam)
        heatmap = overlay_cam(image, cam, alpha=1.0)

        true_label = CIFAR10_CLASSES[labels[i]] if labels[i] is not None else "unknown"
        pred_label = CIFAR10_CLASSES[pred_classes[i].item()]

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(image)
        axes[0].set_title("original")
        axes[1].imshow(heatmap)
        axes[1].set_title("heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("overlay")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"true={true_label} pred={pred_label}")

        out_path = output_dir / f"gradcam_{i:03d}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {vis_images.shape[0]} Grad-CAM panels to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
