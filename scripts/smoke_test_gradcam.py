import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.explain import GradCAM, overlay_cam
from src.models import build_mobilenetv3_small

NUM_CLASSES = 10


def main() -> bool:
    results = []
    torch.manual_seed(0)

    model = build_mobilenetv3_small(variant="vanilla", num_classes=NUM_CLASSES, pretrained=False)
    inputs = torch.randn(4, 3, 224, 224)

    # 1: basic call shape/range checks
    gradcam = GradCAM(model)
    cams, preds = gradcam(inputs)
    results.append(("cams shape is (4, 224, 224)", tuple(cams.shape) == (4, 224, 224)))
    results.append(("cams values <= 1.0", bool((cams <= 1.0 + 1e-6).all())))
    results.append(("cams values >= 0.0 (ReLU)", bool((cams >= 0.0).all())))
    results.append(("preds shape is (4,)", tuple(preds.shape) == (4,)))

    # 2: explicit target_class (int and per-image tensor)
    cams_int, preds_int = gradcam(inputs, target_class=3)
    results.append(("int target_class works, shape ok", tuple(cams_int.shape) == (4, 224, 224)))

    per_image_targets = torch.tensor([0, 1, 2, 3])
    cams_tensor, preds_tensor = gradcam(inputs, target_class=per_image_targets)
    results.append(("per-image tensor target_class works, shape ok", tuple(cams_tensor.shape) == (4, 224, 224)))

    # 3: works inside torch.no_grad()
    with torch.no_grad():
        cams_nograd, preds_nograd = gradcam(inputs)
    results.append(("works inside torch.no_grad()", tuple(cams_nograd.shape) == (4, 224, 224)))

    # 4: overlay_cam shape/dtype
    denorm_image = torch.rand(3, 224, 224)
    overlay = overlay_cam(denorm_image, cams[0])
    results.append(
        (
            "overlay_cam returns HxWx3 uint8",
            overlay.shape == (224, 224, 3) and overlay.dtype.name == "uint8",
        )
    )

    # 5: hook cleanup
    gradcam.remove_hooks()
    no_hooks_after_remove = len(gradcam.target_layer._forward_hooks) == 0
    plain_forward_after_remove_ok = True
    try:
        with torch.no_grad():
            model(inputs)
    except Exception:
        plain_forward_after_remove_ok = False
    results.append(("remove_hooks() leaves target layer hook-free", no_hooks_after_remove))
    results.append(("plain forward pass works after remove_hooks()", plain_forward_after_remove_ok))

    with GradCAM(model) as gradcam_ctx:
        gradcam_ctx(inputs)
    no_hooks_after_ctx = len(gradcam_ctx.target_layer._forward_hooks) == 0
    results.append(("context manager exit removes hooks", no_hooks_after_ctx))

    gradcam_a = GradCAM(model)
    gradcam_b = GradCAM(model)
    hooks_after_two_builds = len(model.features[-1]._forward_hooks)
    gradcam_a.remove_hooks()
    gradcam_b.remove_hooks()
    results.append(("building GradCAM twice registers exactly 2 forward hooks", hooks_after_two_builds == 2))

    # 6: CAM depends on weights
    model_a = build_mobilenetv3_small(variant="vanilla", num_classes=NUM_CLASSES, pretrained=False)
    model_b = copy.deepcopy(model_a)
    for p in model_b.parameters():
        p.data = torch.randn_like(p.data)

    fixed_input = torch.randn(2, 3, 224, 224)
    with GradCAM(model_a) as gc_a:
        cams_a, _ = gc_a(fixed_input)
    with GradCAM(model_b) as gc_b:
        cams_b, _ = gc_b(fixed_input)
    results.append(("CAMs change when weights are randomized", not torch.allclose(cams_a, cams_b)))

    print("\n=== GRAD-CAM SMOKE TEST RESULTS ===")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"[{status}] {name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        import matplotlib.pyplot as plt

        plt.imsave(Path(tmpdir) / "overlay_check.png", overlay)

    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
