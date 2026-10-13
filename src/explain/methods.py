"""Attribution-method registry: one place that maps a `--method` name to an
explainer, so every script/metric that computes a CAM selects it the same way."""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from .gradcam import GradCAM
from .gradcampp import GradCAMPlusPlus
from .random_cam import RandomCAM

METHODS = ("gradcam", "gradcampp", "random")
DEFAULT_METHOD = "gradcam"

METHOD_LABELS = {
    "gradcam": "Grad-CAM",
    "gradcampp": "Grad-CAM++",
    "random": "Random CAM",
}


def build_explainer(
    method: str,
    model: nn.Module,
    target_layer: Optional[nn.Module] = None,
    seed: int = 0,
):
    """Build the explainer for `method`. All share the GradCAM call interface:
    `explainer(input, target_class) -> (cams, pred_classes)`, usable as a context
    manager. `seed` only affects the random baseline (see `RandomCAM`)."""
    if method == "gradcam":
        return GradCAM(model, target_layer=target_layer)
    if method == "gradcampp":
        return GradCAMPlusPlus(model, target_layer=target_layer)
    if method == "random":
        return RandomCAM(model, target_layer=target_layer, seed=seed)
    raise ValueError(f"Unknown attribution method '{method}'. Expected one of {METHODS}.")


def make_cam_fn(
    method: str = DEFAULT_METHOD, seed: int = 0
) -> Callable[[nn.Module, torch.Tensor, torch.device], np.ndarray]:
    """A `cam_fn(model, image, device) -> (H, W) numpy CAM` for the faithfulness metrics."""

    def cam_fn(model: nn.Module, image: torch.Tensor, device: torch.device) -> np.ndarray:
        with build_explainer(method, model, seed=seed) as explainer:
            cams, _ = explainer(image.unsqueeze(0).to(device))
        return cams[0].numpy()

    return cam_fn
