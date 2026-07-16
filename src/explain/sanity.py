from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

from .gradcam import GradCAM


def spearman_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two flattened CAMs. 0.0 if degenerate."""
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()

    if np.allclose(a_flat, a_flat[0]) or np.allclose(b_flat, b_flat[0]):
        return 0.0

    corr, _ = spearmanr(a_flat, b_flat)
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def ssim_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity (data_range=1.0) between two CAMs. 0.0 if degenerate."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    value = structural_similarity(a, b, data_range=1.0)
    if not np.isfinite(value):
        return 0.0
    return float(value)


def randomize_module_(module: nn.Module) -> None:
    """Recursively re-initialize a module's parameters in place, top-down over its subtree."""
    for child in module.children():
        randomize_module_(child)

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        module.reset_parameters()
    elif isinstance(module, nn.BatchNorm2d):
        module.reset_parameters()
        module.reset_running_stats()


def _cascade_order(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Top-down cascade order: classifier module(s) first (last-to-first), then
    the top-level blocks of `features` from last index to first."""
    order: List[Tuple[str, nn.Module]] = []

    if hasattr(model, "classifier"):
        classifier_children = list(model.classifier.children())
        for idx in reversed(range(len(classifier_children))):
            child = classifier_children[idx]
            if any(p.numel() > 0 for p in child.parameters(recurse=True)):
                order.append((f"classifier.{idx}", child))

    if hasattr(model, "features"):
        feature_children = list(model.features.children())
        for idx in reversed(range(len(feature_children))):
            order.append((f"features.{idx}", feature_children[idx]))

    return order


def cascading_randomization(
    model: nn.Module,
    images: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    seed: int = 42,
) -> Tuple[List[Dict], torch.Tensor]:
    """Adebayo et al. (2018) cascading (top-down) parameter-randomization sanity check.

    `model` must already be trained and on the target device; `images` must already be
    on that same device and normalized for the model's input. Returns (steps, reference_cams)
    where `steps` is an ordered list of {step_name, mean_spearman, mean_ssim, cams} records,
    one per cascade step, and `reference_cams` are the CAMs from the intact model.
    """
    model.eval()

    with GradCAM(model, target_layer=target_layer) as gradcam:
        reference_cams, _ = gradcam(images)

    torch.manual_seed(seed)

    order = _cascade_order(model)
    steps: List[Dict] = []

    for step_name, module in order:
        randomize_module_(module)

        with GradCAM(model, target_layer=target_layer) as gradcam:
            cams, _ = gradcam(images)

        spearmans = []
        ssims = []
        for i in range(images.shape[0]):
            ref = reference_cams[i].numpy()
            cur = cams[i].numpy()
            spearmans.append(spearman_similarity(ref, cur))
            ssims.append(ssim_similarity(ref, cur))

        steps.append(
            {
                "step_name": step_name,
                "mean_spearman": float(np.mean(spearmans)),
                "mean_ssim": float(np.mean(ssims)),
                "cams": cams,
            }
        )

    return steps, reference_cams
