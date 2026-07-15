from typing import List, Optional, Tuple, Union

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import get_gradcam_target_layer


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) implemented from scratch with hooks."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else get_gradcam_target_layer(model)

        self._activation: Optional[torch.Tensor] = None
        self._gradient: Optional[torch.Tensor] = None
        self._handles: List = []

        forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._handles.append(forward_handle)

    def _forward_hook(self, module: nn.Module, inputs: Tuple, output: torch.Tensor) -> None:
        self._activation = output
        grad_handle = output.register_hook(self._grad_hook)
        self._handles.append(grad_handle)

    def _grad_hook(self, grad: torch.Tensor) -> None:
        self._gradient = grad

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()

        with torch.enable_grad():
            input_tensor = input_tensor.clone().requires_grad_(True)
            logits = self.model(input_tensor)
            pred_classes = logits.argmax(dim=1)

            if target_class is None:
                score_classes = pred_classes
            elif isinstance(target_class, int):
                score_classes = torch.full_like(pred_classes, target_class)
            else:
                score_classes = target_class.to(logits.device)

            self.model.zero_grad(set_to_none=True)
            selected = logits.gather(1, score_classes.view(-1, 1)).squeeze(1)
            selected.sum().backward()

            activation = self._activation
            gradient = self._gradient
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * activation).sum(dim=1))

            n = cam.shape[0]
            cam_flat = cam.view(n, -1)
            cam_min = cam_flat.min(dim=1, keepdim=True).values
            cam_max = cam_flat.max(dim=1, keepdim=True).values
            cam_range = (cam_max - cam_min).clamp_min(1e-8)
            cam_flat = (cam_flat - cam_min) / cam_range
            cam = cam_flat.view_as(cam)

            h, w = input_tensor.shape[-2:]
            cam = F.interpolate(cam.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)
            cam = cam.squeeze(1)

        return cam.detach().cpu().float(), pred_classes.detach().cpu()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()


def overlay_cam(
    image: Union[torch.Tensor, np.ndarray],
    cam: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()

    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[0] != image.shape[-1]:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    image = np.clip(image, 0.0, 1.0)
    h, w = image.shape[:2]

    colored = cm.get_cmap(colormap)(cam)[..., :3]
    overlay = (1 - alpha) * image + alpha * colored
    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8).reshape(h, w, 3)
