"""Cross-variant Grad-CAM comparison: same test images, multiple checkpoints."""

import random
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data import denormalize
from src.explain.gradcam import GradCAM, overlay_cam

ModelsAndNames = Sequence[Tuple[nn.Module, str]]


def select_shared_indices(
    test_set: Dataset,
    models_and_names: ModelsAndNames,
    device: torch.device,
    num_images: int,
    seed: int = 42,
    only_all_correct: bool = False,
) -> List[int]:
    """Pick a single set of test-set indices to use for every model."""
    n_total = len(test_set)
    rng = random.Random(seed)

    if not only_all_correct:
        return sorted(rng.sample(range(n_total), min(num_images, n_total)))

    correct_indices: Optional[set] = None
    loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
    for model, _name in models_and_names:
        model.eval()
        model_correct = set()
        offset = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                preds = model(images).argmax(dim=1).cpu()
                labels = torch.as_tensor(labels)
                matches = (preds == labels).nonzero(as_tuple=True)[0]
                model_correct.update((offset + matches).tolist())
                offset += images.shape[0]
        correct_indices = model_correct if correct_indices is None else correct_indices & model_correct

    candidates = sorted(correct_indices) if correct_indices else []
    return sorted(rng.sample(candidates, min(num_images, len(candidates))))


def build_comparison(
    models_and_names: ModelsAndNames,
    test_set: Dataset,
    indices: Sequence[int],
    device: torch.device,
    target: str = "true",
) -> List[Dict]:
    """Run every model's Grad-CAM over the same shared images."""
    if target not in ("true", "pred"):
        raise ValueError(f"target must be 'true' or 'pred', got {target!r}")
    if len(indices) == 0:
        return []

    images = torch.stack([test_set[i][0] for i in indices]).to(device)
    true_labels = [test_set[i][1] for i in indices]

    per_model_results = []
    for model, name in models_and_names:
        model.eval()
        with torch.no_grad():
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            pred_classes = logits.argmax(dim=1)
            confidences = probs.gather(1, pred_classes.view(-1, 1)).squeeze(1)

        target_class = (
            torch.tensor(true_labels, device=device, dtype=torch.long) if target == "true" else None
        )
        with GradCAM(model) as gradcam:
            cams, _ = gradcam(images, target_class=target_class)

        per_model_results.append(
            {
                "name": name,
                "pred_classes": pred_classes.cpu(),
                "confidences": confidences.cpu(),
                "cams": cams,
            }
        )

    results = []
    for row, idx in enumerate(indices):
        original_image = denormalize(test_set[idx][0]).permute(1, 2, 0).numpy()
        per_model = [
            {
                "name": pm["name"],
                "pred_label": int(pm["pred_classes"][row]),
                "confidence": float(pm["confidences"][row]),
                "cam": pm["cams"][row].numpy(),
            }
            for pm in per_model_results
        ]
        results.append(
            {
                "index": idx,
                "true_label": true_labels[row],
                "original_image": original_image,
                "per_model": per_model,
            }
        )
    return results


def render_comparison_grid(result: List[Dict], class_names: Sequence[str], out_path) -> None:
    """One row per image: [Original | overlay per model]. Saves a single PNG."""
    n_images = len(result)
    n_models = len(result[0]["per_model"]) if n_images > 0 else 0
    n_cols = 1 + n_models

    fig, axes = plt.subplots(
        max(n_images, 1), n_cols, figsize=(2.4 * n_cols, 2.4 * max(n_images, 1)), squeeze=False
    )

    for row, item in enumerate(result):
        image = item["original_image"]
        true_label = class_names[item["true_label"]]

        axes[row][0].imshow(image)
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        axes[row][0].set_ylabel(true_label, fontsize=9)
        if row == 0:
            axes[row][0].set_title("Original", fontsize=10)

        for col, pm in enumerate(item["per_model"], start=1):
            overlay = overlay_cam(image, pm["cam"])
            axes[row][col].imshow(overlay)
            axes[row][col].axis("off")

            wrong = pm["pred_label"] != item["true_label"]
            subtitle = f"{class_names[pm['pred_label']]} ({pm['confidence']:.2f})"
            if wrong:
                subtitle += " (WRONG)"
            color = "red" if wrong else "black"

            title = f"{pm['name']}\n{subtitle}" if row == 0 else subtitle
            axes[row][col].set_title(title, fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
