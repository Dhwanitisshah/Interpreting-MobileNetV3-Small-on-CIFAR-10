import hashlib
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# torch.Generator.manual_seed accepts up to 2**63 - 1.
_SEED_MASK = 0x7FFFFFFFFFFFFFFF


class RandomCAM:
    """Null-baseline "explainer": a uniform-noise CAM in [0, 1] at input resolution.

    Matches the `GradCAM` call interface (`__call__(input, target_class)` returns
    `(cams, pred_classes)`, plus `remove_hooks` and the context-manager protocol)
    so it drops in anywhere a CAM is computed, but ignores the model's weights
    and the target class entirely. It is the floor a real attribution method
    must beat on every metric (deletion/insertion/ROAD, drift).

    "Fixed-seeded" means each image's noise map is a pure function of
    (base `seed`, the image's own bytes): the same image always gets the same
    map -- across calls, models, training seeds and evaluation order -- so the
    baseline is reproducible and paired across every cell of a campaign. Two
    different images (e.g. a clean image and its corrupted version) get
    independent maps, which is exactly what the drift metric should see for a
    random explanation (maximal drift).

    Consequence for the cascading-randomization sanity check: the maps do not
    depend on the model, so they never change as weights are randomized
    (similarity stays at 1.0 at every step). That is the signature of a
    model-independent method and is the intended result for this baseline.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, seed: int = 0):
        # `target_layer` is accepted (and ignored) for interface parity with GradCAM.
        self.model = model
        self.target_layer = target_layer
        self.seed = int(seed)

    def _image_seed(self, image: torch.Tensor) -> int:
        data = image.detach().cpu().contiguous().numpy().tobytes()
        digest = hashlib.blake2b(data, digest_size=8).digest()
        return (int.from_bytes(digest, "little") ^ self.seed) & _SEED_MASK

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            pred_classes = self.model(input_tensor).argmax(dim=1)

        h, w = input_tensor.shape[-2:]
        cams = torch.stack(
            [
                torch.rand(h, w, generator=torch.Generator().manual_seed(self._image_seed(image)))
                for image in input_tensor
            ]
        )
        return cams, pred_classes.detach().cpu()

    def remove_hooks(self) -> None:
        pass

    def __enter__(self) -> "RandomCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()
