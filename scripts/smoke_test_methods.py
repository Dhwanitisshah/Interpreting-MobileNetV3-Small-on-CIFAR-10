"""Smoke test for the Phase 8.2 attribution methods (Grad-CAM++ and the random-CAM
null baseline) and their wiring through faithfulness, drift and the sanity check.
No download. Verifies:

  - Grad-CAM++ passes the same shape/range/interface checks as Grad-CAM, its
    channel weights match a hand-computed value of the closed-form formula, and
    it is genuinely a different map from Grad-CAM;
  - the random CAM is uniform noise in [0, 1], deterministic (fixed-seeded),
    model- and class-independent, and independent between different images;
  - on a small CNN actually TRAINED on a localized-signal toy task, both real
    methods beat the random baseline on deletion, insertion and ROAD;
  - evaluate_robustness / cascading_randomization accept every method, and the
    random baseline shows maximal drift and (being model-independent) no
    sanity-check decay;
  - the --method flag is wired into the three eval scripts (default gradcam).
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/

import numpy as np
import torch
import torch.nn as nn

from src.explain import (
    METHODS,
    GradCAM,
    GradCAMPlusPlus,
    RandomCAM,
    build_explainer,
    cascading_randomization,
    make_cam_fn,
    spearman_similarity,
)
from src.metrics import evaluate_model_faithfulness
from src.models import build_mobilenetv3_small
from src.robustness import evaluate_robustness
from src.utils import set_seed

from smoke_test_robustness import SyntheticDataset, TinyModel

NUM_CLASSES = 10
TOY_SIZE = 32


class ToyCNN(nn.Module):
    """Small CNN with a `features` Sequential, so the default Grad-CAM target
    layer (`features[-1]`) applies to it just as it does to MobileNetV3."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


def make_toy_batch(n: int, generator: torch.Generator, blob: bool) -> torch.Tensor:
    """Class 1 = an 8x8 bright blob at a random location on noise; class 0 = noise only.
    The blob is the ONLY class-relevant signal, so a faithful CAM must find it."""
    x = 0.3 * torch.randn(n, 3, TOY_SIZE, TOY_SIZE, generator=generator)
    if blob:
        for i in range(n):
            r, c = (int(v) for v in torch.randint(0, TOY_SIZE - 8, (2,), generator=generator))
            x[i, :, r : r + 8, c : c + 8] += 3.0
    return x


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, images: torch.Tensor, label: int):
        self.images, self.label = images, label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.label


def train_toy_model() -> ToyCNN:
    set_seed(0)
    g = torch.Generator().manual_seed(0)
    model = ToyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(150):
        x = torch.cat([make_toy_batch(16, g, blob=True), make_toy_batch(16, g, blob=False)])
        y = torch.cat([torch.ones(16, dtype=torch.long), torch.zeros(16, dtype=torch.long)])
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    return model


def parse_with(module_name: str, argv: list):
    module = __import__(module_name)
    old_argv = sys.argv
    sys.argv = [f"{module_name}.py"] + argv
    try:
        return module.parse_args()
    finally:
        sys.argv = old_argv


def main() -> bool:
    results = []
    torch.manual_seed(0)

    # ---------- 1: Grad-CAM++ matches Grad-CAM's interface and validity checks ----------
    model = build_mobilenetv3_small(variant="vanilla", num_classes=NUM_CLASSES, pretrained=False)
    inputs = torch.randn(4, 3, 224, 224)

    with GradCAM(model) as gc:
        cams_gc, preds_gc = gc(inputs)
    gcpp = GradCAMPlusPlus(model)
    cams, preds = gcpp(inputs)
    results.append(("GradCAMPlusPlus is-a GradCAM (same interface/target layer)", isinstance(gcpp, GradCAM)))
    results.append(("Grad-CAM++ cams shape is (4, 224, 224)", tuple(cams.shape) == (4, 224, 224)))
    results.append(("Grad-CAM++ cams are finite", bool(torch.isfinite(cams).all())))
    results.append(("Grad-CAM++ cams values <= 1.0", bool((cams <= 1.0 + 1e-6).all())))
    results.append(("Grad-CAM++ cams values >= 0.0 (ReLU)", bool((cams >= 0.0).all())))
    results.append(("Grad-CAM++ preds shape is (4,) and equal Grad-CAM's", tuple(preds.shape) == (4,) and bool((preds == preds_gc).all())))
    results.append(("Grad-CAM++ uses the same target layer as Grad-CAM", gcpp.target_layer is gc.target_layer))

    cams_int, _ = gcpp(inputs, target_class=3)
    cams_tensor, _ = gcpp(inputs, target_class=torch.tensor([0, 1, 2, 3]))
    results.append(("Grad-CAM++ int / per-image tensor target_class work, shape ok",
                    tuple(cams_int.shape) == (4, 224, 224) and tuple(cams_tensor.shape) == (4, 224, 224)))
    results.append(("Grad-CAM++ depends on the target class", not torch.allclose(cams_int, cams_tensor)))
    with torch.no_grad():
        cams_nograd, _ = gcpp(inputs)
    results.append(("Grad-CAM++ works inside torch.no_grad()", tuple(cams_nograd.shape) == (4, 224, 224)))
    results.append(("Grad-CAM++ is a different map from Grad-CAM", not torch.allclose(cams, cams_gc)))

    rho = float(np.mean([spearman_similarity(cams[i].numpy(), cams_gc[i].numpy()) for i in range(4)]))
    results.append((f"Grad-CAM++ and Grad-CAM agree in rank on average (mean Spearman {rho:.2f} > 0)", rho > 0.0))

    gcpp.remove_hooks()
    results.append(("Grad-CAM++ remove_hooks() leaves target layer hook-free", len(gcpp.target_layer._forward_hooks) == 0))

    model_b = copy.deepcopy(model)
    for p in model_b.parameters():
        p.data = torch.randn_like(p.data)
    with GradCAMPlusPlus(model) as a, GradCAMPlusPlus(model_b) as b:
        cams_a, _ = a(inputs[:2])
        cams_b, _ = b(inputs[:2])
    results.append(("Grad-CAM++ CAMs change when weights are randomized", not torch.allclose(cams_a, cams_b)))

    # closed-form check on a hand-computed 1-channel 2x2 case:
    #   g = [[1, 2], [0, -1]], A = [[1, 1], [2, 3]] -> sum(A) = 7
    #   alpha = g^2 / (2 g^2 + 7 g^3) = [[1/9, 1/16], [0 (zero denom -> 0), -1/5]]
    #   w = sum(alpha * ReLU(g)) = 1/9 * 1 + 1/16 * 2
    g_hand = torch.tensor([[[[1.0, 2.0], [0.0, -1.0]]]])
    a_hand = torch.tensor([[[[1.0, 1.0], [2.0, 3.0]]]])
    formula_probe = GradCAMPlusPlus(model)
    w_hand = float(formula_probe._channel_weights(a_hand, g_hand))
    results.append((f"Grad-CAM++ weights match the hand-computed formula ({w_hand:.6f} vs {1 / 9 + 2 / 16:.6f})",
                    abs(w_hand - (1 / 9 + 2 / 16)) < 1e-6))
    # all-zero gradient must not produce NaN (zero denominators handled)
    w_zero = formula_probe._channel_weights(a_hand, torch.zeros_like(a_hand))
    results.append(("Grad-CAM++ weights are finite (== 0) for an all-zero gradient", bool(torch.isfinite(w_zero).all()) and float(w_zero) == 0.0))
    formula_probe.remove_hooks()

    # ---------- 2: random CAM ----------
    rc = RandomCAM(model, seed=42)
    rcams, rpreds = rc(inputs)
    results.append(("random cams shape is (4, 224, 224)", tuple(rcams.shape) == (4, 224, 224)))
    results.append(("random cams are within [0, 1]", bool((rcams >= 0.0).all() and (rcams <= 1.0).all())))
    results.append(("random cams are uniform noise (mean ~0.5, std ~0.289)",
                    abs(float(rcams.mean()) - 0.5) < 0.01 and abs(float(rcams.std()) - 0.2887) < 0.01))
    results.append(("random preds are the model's argmax predictions", bool((rpreds == preds_gc).all())))
    results.append(("random cams are deterministic: same image, same map (across calls and instances)",
                    bool(torch.equal(rcams, rc(inputs)[0])) and bool(torch.equal(rcams, RandomCAM(model, seed=42)(inputs)[0]))))
    results.append(("random cams are batch-composition independent (image alone == image in batch)",
                    bool(torch.equal(rc(inputs[1:2])[0][0], rcams[1]))))
    results.append(("random cams differ between different images", not torch.equal(rcams[0], rcams[1])))
    results.append(("random cams change with the base seed", not torch.equal(rcams, RandomCAM(model, seed=43)(inputs)[0])))
    results.append(("random cams ignore the target class", bool(torch.equal(rcams, rc(inputs, target_class=7)[0]))))
    results.append(("random cams ignore the model's weights", bool(torch.equal(rcams, RandomCAM(model_b, seed=42)(inputs)[0]))))

    # ---------- factory ----------
    built = [build_explainer(m, model) for m in METHODS]
    results.append(("build_explainer returns the right class per method",
                    [type(e) for e in built] == [GradCAM, GradCAMPlusPlus, RandomCAM]))
    for explainer in built:
        explainer.remove_hooks()
    results.append(("no explainer leaks a hook onto the model after remove_hooks()",
                    len(model.features[-1]._forward_hooks) == 0))
    raised = False
    try:
        build_explainer("scorecam", model)
    except ValueError:
        raised = True
    results.append(("build_explainer rejects an unknown method with ValueError", raised))
    results.append(("make_cam_fn returns a (H, W) numpy CAM for every method",
                    all(make_cam_fn(m, seed=0)(model, inputs[0], torch.device("cpu")).shape == (224, 224) for m in METHODS)))

    # ---------- 3: real vs random on a TRAINED toy model (faithfulness) ----------
    toy = train_toy_model()
    g_test = torch.Generator().manual_seed(123)
    test_images = make_toy_batch(16, g_test, blob=True)
    with torch.no_grad():
        toy_acc = float((toy(test_images).argmax(1) == 1).float().mean())
    results.append((f"toy model trained to classify the blob task (test acc {toy_acc:.0%} >= 90%)", toy_acc >= 0.9))

    # min-max normalized at feature-map resolution, then bilinearly upsampled, so the peak
    # lands just under 1.0 (checked on the trained model: a randomly initialised network's
    # raw CAM can fall below the normalizer's 1e-8 range clamp, which is not meaningful)
    for method in ("gradcam", "gradcampp"):
        with build_explainer(method, toy) as explainer:
            toy_cams, _ = explainer(test_images)
        results.append((f"{method}: trained-model cams are normalized per image (peak > 0.9, min == 0)",
                        bool((toy_cams.flatten(1).max(1).values > 0.9).all() and (toy_cams.flatten(1).min(1).values == 0).all())))

    dataset = ToyDataset(test_images, label=1)
    indices = list(range(len(dataset)))
    agg = {}
    for method in METHODS:
        _, agg[method] = evaluate_model_faithfulness(
            toy, dataset, indices, torch.device("cpu"), make_cam_fn(method, seed=42), steps=10,
            insertion_baseline="mean",  # decisive on the blob task; "blur" leaves a blurred blob that is still class 1
            desc=f"toy[{method}]",
        )
    print("\n[toy trained model] mean normalized faithfulness per method")
    for method in METHODS:
        a = agg[method]
        print(f"  {method:<10} deletion={a['deletion_auc_mean']:.3f}  insertion={a['insertion_auc_mean']:.3f}  road_gap={a['road_gap_mean']:.3f}")

    for real in ("gradcam", "gradcampp"):
        results.append((f"{real}: deletion AUC lower (better) than random", agg[real]["deletion_auc_mean"] < agg["random"]["deletion_auc_mean"]))
        results.append((f"{real}: insertion AUC higher (better) than random", agg[real]["insertion_auc_mean"] > agg["random"]["insertion_auc_mean"]))
        results.append((f"{real}: ROAD gap higher (better) than random", agg[real]["road_gap_mean"] > agg["random"]["road_gap_mean"]))

    # ---------- 4: drift under every method ----------
    dataset_224 = SyntheticDataset(n=3, seed=0)
    drift = {}
    for method in METHODS:
        set_seed(0)
        records, aggregate = evaluate_robustness(
            TinyModel(), dataset_224, [0, 1, 2], torch.device("cpu"), ["brightness"], [1],
            desc=f"drift[{method}]", method=method, method_seed=42,
        )
        set_seed(0)
        records_again, _ = evaluate_robustness(
            TinyModel(), dataset_224, [0, 1, 2], torch.device("cpu"), ["brightness"], [1],
            desc=f"drift[{method}]", method=method, method_seed=42,
        )
        drift[method] = float(np.mean([r["spearman"] for r in records]))
        results.append((f"evaluate_robustness(method={method}) returns one record per image/corruption/severity", len(records) == 3))
        results.append((f"evaluate_robustness(method={method}) is bitwise reproducible", records == records_again))
    print("\n[drift, brightness sev1] mean Spearman(clean CAM, corrupted CAM): " + ", ".join(f"{m}={v:.3f}" for m, v in drift.items()))
    results.append(("random baseline has ~zero CAM agreement under corruption (maximal drift)", abs(drift["random"]) < 0.05))
    results.append(("real methods drift less than the random baseline", drift["gradcam"] > drift["random"] + 0.1 and drift["gradcampp"] > drift["random"] + 0.1))

    # ---------- 5: sanity check under every method ----------
    sanity_images = torch.rand(2, 3, 224, 224)
    for method in METHODS:
        fresh = build_mobilenetv3_small(variant="vanilla", num_classes=NUM_CLASSES, pretrained=False)
        steps, ref = cascading_randomization(fresh, sanity_images, seed=42, method=method)
        results.append((f"cascading_randomization(method={method}) returns steps and (2, 224, 224) reference cams",
                        len(steps) > 0 and tuple(ref.shape) == (2, 224, 224)))
        final = steps[-1]["mean_spearman"]
        if method == "random":
            flat = all(abs(s["mean_spearman"] - 1.0) < 1e-6 and abs(s["mean_ssim"] - 1.0) < 1e-6 for s in steps)
            results.append(("random baseline never decays under randomization (model-independent: sim == 1.0 at every step)", flat))
        else:
            results.append((f"{method}: final fully-randomized step differs from the reference (Spearman {final:.2f} < 1)", final < 0.999))

    # ---------- 6: --method wired into the three eval scripts ----------
    wired = True
    for module_name, base in (
        ("faithfulness_eval", ["--checkpoints", "x.pth"]),
        ("robustness_eval", ["--checkpoints", "x.pth"]),
        ("sanity_check", ["--checkpoint", "x.pth"]),
    ):
        default_ok = parse_with(module_name, base).method == "gradcam"
        explicit_ok = all(parse_with(module_name, base + ["--method", m]).method == m for m in METHODS)
        try:
            parse_with(module_name, base + ["--method", "bogus"])
            rejects = False
        except SystemExit:
            rejects = True
        wired = wired and default_ok and explicit_ok and rejects
        results.append((f"{module_name}.py: --method defaults to gradcam, accepts {list(METHODS)}, rejects others", default_ok and explicit_ok and rejects))

    # ---------- 7: concentration_diagnostic refuses non-Grad-CAM robustness output ----------
    import concentration_diagnostic

    def guard_raises(metrics: dict) -> bool:
        try:
            concentration_diagnostic.check_grad_cam_only(metrics)
            return False
        except ValueError:
            return True

    results.append(("concentration_diagnostic guard: rejects gradcampp/random, accepts gradcam and legacy (no 'method')",
                    guard_raises({"m": {"method": "gradcampp"}}) and guard_raises({"m": {"method": "random"}})
                    and not guard_raises({"m": {"method": "gradcam"}}) and not guard_raises({"m": {}})))

    print("\n=== ATTRIBUTION-METHODS SMOKE TEST RESULTS ===")
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
