import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation

from src.data import denormalize
from src.models import VARIANTS, build_mobilenetv3_small
from src.utils import load_config
from src.utils.seed import set_seed

EXPECTED_SE = {"vanilla": 9, "no_se": 0, "small_kernel": 9}
EXPECTED_5X5 = {"vanilla": 8, "no_se": 8, "small_kernel": 0}


def count_modules(model: nn.Module) -> tuple:
    se_count = sum(1 for m in model.modules() if isinstance(m, SqueezeExcitation))
    conv5x5_count = sum(
        1
        for m in model.modules()
        if isinstance(m, nn.Conv2d) and m.kernel_size == (5, 5)
    )
    return se_count, conv5x5_count


def main() -> bool:
    set_seed(42)
    results = []

    # 1 & 2: build variants, forward pass, count SE / 5x5 modules
    for variant in VARIANTS:
        model = build_mobilenetv3_small(variant=variant, num_classes=10, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 224, 224))
        shape_ok = out.shape == (2, 10)
        se_count, conv5x5_count = count_modules(model)
        se_ok = se_count == EXPECTED_SE[variant]
        conv_ok = conv5x5_count == EXPECTED_5X5[variant]
        results.append(
            (
                f"[{variant}] forward pass shape {tuple(out.shape)} == (2, 10)",
                shape_ok,
            )
        )
        results.append(
            (
                f"[{variant}] SE blocks = {se_count} (expected {EXPECTED_SE[variant]})",
                se_ok,
            )
        )
        results.append(
            (
                f"[{variant}] 5x5 depthwise convs = {conv5x5_count} (expected {EXPECTED_5X5[variant]})",
                conv_ok,
            )
        )

    # 3: no_se + pretrained=True must raise ValueError
    try:
        build_mobilenetv3_small(variant="no_se", pretrained=True)
        raised = False
    except ValueError:
        raised = True
    results.append(("build_mobilenetv3_small('no_se', pretrained=True) raises ValueError", raised))

    # 4: denormalize(normalize(img)) ~= img
    from torchvision.transforms import Normalize

    from src.data.cifar10 import IMAGENET_MEAN, IMAGENET_STD

    img = torch.rand(3, 224, 224)
    normalize_only = Normalize(IMAGENET_MEAN, IMAGENET_STD)
    normalized_img = normalize_only(img)
    recovered = denormalize(normalized_img)
    denorm_ok = torch.allclose(recovered, img, atol=1e-5)
    results.append(("denormalize(normalize(img)) ~= img", denorm_ok))

    # 5: config dot-access
    config_path = Path(__file__).resolve().parent.parent / "configs" / "no_se_scratch.yaml"
    cfg = load_config(config_path)
    cfg_ok = cfg.model.variant == "no_se"
    results.append(("configs/no_se_scratch.yaml dot-access cfg.model.variant == 'no_se'", cfg_ok))

    print("\n=== SMOKE TEST RESULTS ===")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"[{status}] {name}")

    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
