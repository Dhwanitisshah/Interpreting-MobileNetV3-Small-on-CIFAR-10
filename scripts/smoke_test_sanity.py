import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.explain import cascading_randomization, randomize_module_, spearman_similarity, ssim_similarity
from src.explain.sanity import _cascade_order
from src.models import build_mobilenetv3_small

NUM_CLASSES = 10


def main() -> bool:
    results = []
    torch.manual_seed(0)

    model = build_mobilenetv3_small(variant="vanilla", num_classes=NUM_CLASSES, pretrained=False)
    images = torch.rand(3, 3, 224, 224)

    # 1: similarity helpers, identity case
    rng = np.random.default_rng(0)
    x = rng.random((224, 224)).astype(np.float64)
    spearman_self = spearman_similarity(x, x)
    ssim_self = ssim_similarity(x, x)
    results.append(("spearman_similarity(x, x) == 1.0", abs(spearman_self - 1.0) < 1e-6))
    results.append(("ssim_similarity(x, x) ~= 1.0", abs(ssim_self - 1.0) < 1e-6))

    # 2: cascading_randomization structure
    expected_order = _cascade_order(model)
    steps, reference_cams = cascading_randomization(model, images, seed=42)

    results.append(("cascading_randomization returns a non-empty ordered list", len(steps) > 0))
    step_names = [s["step_name"] for s in steps]
    results.append(("step_names are unique", len(step_names) == len(set(step_names))))
    results.append(
        (
            "num steps == (#classifier modules randomized + #feature blocks)",
            len(steps) == len(expected_order),
        )
    )
    results.append(("reference_cams shape is (3, 224, 224)", tuple(reference_cams.shape) == (3, 224, 224)))

    # 3: per-step metrics are finite and in range
    metrics_ok = True
    for s in steps:
        ms, mssim = s["mean_spearman"], s["mean_ssim"]
        if not (np.isfinite(ms) and -1.0 - 1e-6 <= ms <= 1.0 + 1e-6):
            metrics_ok = False
        if not (np.isfinite(mssim) and -1.0 - 1e-6 <= mssim <= 1.0 + 1e-6):
            metrics_ok = False
    results.append(("each step's mean_spearman/mean_ssim are finite and in range", metrics_ok))

    # 4: final (fully random) step differs from the reference
    final_spearman = steps[-1]["mean_spearman"]
    results.append(("final step mean_spearman < 1.0 (randomization changed the CAMs)", final_spearman < 1.0))

    # 5: randomize_module_ actually changes Conv2d weights
    conv = torch.nn.Conv2d(3, 8, kernel_size=3)
    weight_before = conv.weight.detach().clone()
    randomize_module_(conv)
    weight_after = conv.weight.detach()
    results.append(("randomize_module_ changes Conv2d weights", not torch.allclose(weight_before, weight_after)))

    print("\n=== SANITY CHECK SMOKE TEST RESULTS ===")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"[{status}] {name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        import json

        summary = [{"step_name": s["step_name"], "mean_spearman": s["mean_spearman"], "mean_ssim": s["mean_ssim"]} for s in steps]
        with open(Path(tmpdir) / "sanity_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
