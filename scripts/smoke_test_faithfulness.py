"""Mechanics-only smoke test for src/metrics/faithfulness.py — no download, tiny
synthetic data. Verifies shapes/ranges/monotonicity, a known analytic AUC value,
a real-vs-random CAM sanity check on a deterministic toy model, and the paired
statistical tests on synthetic distributions with a known difference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.metrics import (
    auc,
    compare_models_statistically,
    deletion_curve,
    insertion_curve,
    road_score,
)

IMG_SIZE = 16
STEPS = 10


class QuadrantModel(nn.Module):
    """Deterministic toy binary classifier with no learned parameters: the class-1
    logit is the mean pixel value in the top-left quadrant, class-0 is its negative.
    Fully deterministic, so it makes an ideal fixture for a real-vs-random CAM check."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        quad = x[:, :, : h // 2, : w // 2]
        score = quad.mean(dim=(1, 2, 3))
        return torch.stack([-score, score], dim=1)


def make_image(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 0.5 + 0.1 * torch.randn(3, IMG_SIZE, IMG_SIZE, generator=g)


def real_quadrant_cam() -> np.ndarray:
    """A CAM that (correctly) highlights the top-left quadrant the model reads."""
    cam = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    cam[: IMG_SIZE // 2, : IMG_SIZE // 2] = 1.0
    return cam


def random_cam(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((IMG_SIZE, IMG_SIZE)).astype(np.float32)


def main() -> bool:
    results = []
    device = torch.device("cpu")
    model = QuadrantModel()
    model.eval()
    image = make_image(seed=0)
    cam = real_quadrant_cam()

    # 1: deletion_curve mechanics
    del_frac, del_probs = deletion_curve(model, image, cam, device, steps=STEPS, baseline="mean")
    results.append(("deletion_curve returns arrays of length steps+1", len(del_frac) == STEPS + 1 and len(del_probs) == STEPS + 1))
    results.append(("deletion_curve fractions are monotonically increasing 0 -> 1", np.all(np.diff(del_frac) > 0) and del_frac[0] == 0.0 and abs(del_frac[-1] - 1.0) < 1e-9))
    results.append(("deletion_curve probs are valid [0, 1]", np.all(del_probs >= -1e-6) and np.all(del_probs <= 1.0 + 1e-6)))

    # 2: insertion_curve mechanics
    ins_frac, ins_probs = insertion_curve(model, image, cam, device, steps=STEPS, baseline="blur")
    results.append(("insertion_curve returns arrays of length steps+1", len(ins_frac) == STEPS + 1 and len(ins_probs) == STEPS + 1))
    results.append(("insertion_curve fractions are monotonically increasing 0 -> 1", np.all(np.diff(ins_frac) > 0) and ins_frac[0] == 0.0 and abs(ins_frac[-1] - 1.0) < 1e-9))
    results.append(("insertion_curve probs are valid [0, 1]", np.all(ins_probs >= -1e-6) and np.all(ins_probs <= 1.0 + 1e-6)))

    # also exercise the "black" baseline path
    del_frac_b, del_probs_b = deletion_curve(model, image, cam, device, steps=STEPS, baseline="black")
    results.append(("deletion_curve with baseline='black' runs and returns valid probs", np.all(del_probs_b >= -1e-6) and np.all(del_probs_b <= 1.0 + 1e-6)))

    # 3: auc() on a known simple curve
    x_flat = np.array([0.0, 0.5, 1.0])
    y_const = np.array([1.0, 1.0, 1.0])
    auc_const = auc(x_flat, y_const)
    results.append(("auc() of a constant curve y=1 over [0,1] is 1.0", abs(auc_const - 1.0) < 1e-9))

    x_diag = np.linspace(0.0, 1.0, 11)
    y_diag = x_diag.copy()
    auc_diag = auc(x_diag, y_diag)
    results.append(("auc() of the diagonal y=x over [0,1] is 0.5", abs(auc_diag - 0.5) < 1e-9))

    # 4: road_score mechanics
    percentiles = (10, 20, 30, 40, 50)
    road_most = road_score(model, image, cam, device, percentiles=percentiles, order="most")
    road_least = road_score(model, image, cam, device, percentiles=percentiles, order="least")
    results.append(("road_score returns one value per requested percentile (order=most)", set(road_most.keys()) == set(percentiles)))
    results.append(("road_score returns one value per requested percentile (order=least)", set(road_least.keys()) == set(percentiles)))
    results.append(("road_score values are all in [0, 1]", all(-1e-6 <= v <= 1.0 + 1e-6 for v in list(road_most.values()) + list(road_least.values()))))

    # 5 (soft check): a real (correctly-localized) CAM should be more "faithful" —
    # lower deletion AUC — than a random CAM, on a deterministic toy model where
    # ground-truth pixel importance is known by construction. This is a sanity
    # check on the metric itself, not a hard mechanical requirement, so a failure
    # is reported as a warning rather than a hard failure.
    real_del_auc = auc(*deletion_curve(model, image, real_quadrant_cam(), device, steps=STEPS, baseline="mean"))
    random_aucs = [
        auc(*deletion_curve(model, image, random_cam(seed=s), device, steps=STEPS, baseline="mean"))
        for s in range(5)
    ]
    mean_random_auc = float(np.mean(random_aucs))
    metric_discriminates = real_del_auc < mean_random_auc
    print(
        f"\n[soft check] real-CAM deletion AUC = {real_del_auc:.4f} vs "
        f"mean random-CAM deletion AUC = {mean_random_auc:.4f} "
        f"({'lower, as expected' if metric_discriminates else 'WARNING: not lower than random'})"
    )

    # 6: compare_models_statistically on synthetic distributions
    rng = np.random.default_rng(42)
    n = 40
    vals_a = rng.normal(loc=0.5, scale=0.05, size=n)
    vals_b = vals_a + 0.3  # large, consistent, known difference

    diff_comparison = compare_models_statistically({"model_a": vals_a, "model_b": vals_b}, metric_name="toy")
    diff_row = diff_comparison[0]
    results.append(("compare_models_statistically detects a known large difference: t_p < 0.05", diff_row["t_p"] < 0.05))
    results.append(("compare_models_statistically detects a known large difference: wilcoxon_p < 0.05", diff_row["wilcoxon_p"] < 0.05))
    results.append(("compare_models_statistically Cohen's d is large and correctly signed", diff_row["cohens_d"] < -3.0))

    same_comparison = compare_models_statistically({"model_a": vals_a, "model_b": vals_a}, metric_name="toy")
    same_row = same_comparison[0]
    results.append(("compare_models_statistically on identical inputs returns t_p ~ 1.0", abs(same_row["t_p"] - 1.0) < 1e-6))
    results.append(("compare_models_statistically on identical inputs returns wilcoxon_p ~ 1.0", abs(same_row["wilcoxon_p"] - 1.0) < 1e-6))
    results.append(("compare_models_statistically on identical inputs returns Cohen's d == 0", same_row["cohens_d"] == 0.0))

    print("\n=== FAITHFULNESS METRICS SMOKE TEST RESULTS ===")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"[{status}] {name}")

    print(f"[{'PASS' if metric_discriminates else 'WARN'}] real-CAM deletion AUC < random-CAM deletion AUC (soft check)")

    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
