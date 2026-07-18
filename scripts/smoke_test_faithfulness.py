"""Mechanics-only smoke test for src/metrics/faithfulness.py — no download, tiny
synthetic data. Verifies shapes/ranges/monotonicity, a known analytic AUC value,
a real-vs-random CAM sanity check on a deterministic toy model, confidence
normalization (curves start at 1.0, and are scale-invariant across models with
different baseline confidence but identical pixel-importance ranking), strict
checkpoint loading failing loudly on a mismatch, and the paired statistical
tests (including effect-size labeling and Bonferroni correction) on synthetic
distributions with a known difference.
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
    effect_size_label,
    insertion_curve,
    road_score,
)
from src.models import build_mobilenetv3_small

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


class ScaledQuadrantModel(nn.Module):
    """Deterministic toy binary classifier: P(class 1) = k * s(image), where s is a
    fixed underlying "importance" function (top-left quadrant mean, clamped to
    [0, 1]) shared by every k. Different k values simulate models with different
    overall confidence but an IDENTICAL underlying pixel-importance ranking and
    curve shape — the fixture for testing that P/P0 normalization exactly removes
    the confidence-scale effect. k must be chosen so k*s stays in (0, 1) over the
    image's value range and so k*s(original) > 0.5 (both models must agree class 1
    is the argmax on the unperturbed image, or they'd be tracking different classes)."""

    def __init__(self, k: float):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        quad = x[:, :, : h // 2, : w // 2]
        s = quad.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
        p1 = (self.k * s).clamp(1e-6, 1 - 1e-6)
        logit1 = torch.log(p1 / (1 - p1))
        logit0 = torch.zeros_like(logit1)
        return torch.stack([logit0, logit1], dim=1)


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
    del_frac, del_probs, del_p0 = deletion_curve(model, image, cam, device, steps=STEPS, baseline="mean")
    results.append(("deletion_curve returns arrays of length steps+1", len(del_frac) == STEPS + 1 and len(del_probs) == STEPS + 1))
    results.append(("deletion_curve fractions are monotonically increasing 0 -> 1", np.all(np.diff(del_frac) > 0) and del_frac[0] == 0.0 and abs(del_frac[-1] - 1.0) < 1e-9))
    results.append(("deletion_curve probs are valid [0, 1]", np.all(del_probs >= -1e-6) and np.all(del_probs <= 1.0 + 1e-6)))
    results.append(("deletion_curve p0 equals probs[0]", abs(del_p0 - del_probs[0]) < 1e-6))

    # 2: insertion_curve mechanics
    ins_frac, ins_probs, ins_p0 = insertion_curve(model, image, cam, device, steps=STEPS, baseline="blur")
    results.append(("insertion_curve returns arrays of length steps+1", len(ins_frac) == STEPS + 1 and len(ins_probs) == STEPS + 1))
    results.append(("insertion_curve fractions are monotonically increasing 0 -> 1", np.all(np.diff(ins_frac) > 0) and ins_frac[0] == 0.0 and abs(ins_frac[-1] - 1.0) < 1e-9))
    results.append(("insertion_curve probs are valid [0, 1]", np.all(ins_probs >= -1e-6) and np.all(ins_probs <= 1.0 + 1e-6)))
    results.append(("insertion_curve p0 matches deletion_curve's p0 (same image, same target class)", abs(ins_p0 - del_p0) < 1e-6))

    # also exercise the "black" baseline path
    del_frac_b, del_probs_b, _ = deletion_curve(model, image, cam, device, steps=STEPS, baseline="black")
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
    road_most, road_most_p0 = road_score(model, image, cam, device, percentiles=percentiles, order="most")
    road_least, road_least_p0 = road_score(model, image, cam, device, percentiles=percentiles, order="least")
    results.append(("road_score returns one value per requested percentile (order=most)", set(road_most.keys()) == set(percentiles)))
    results.append(("road_score returns one value per requested percentile (order=least)", set(road_least.keys()) == set(percentiles)))
    results.append(("road_score values are all in [0, 1]", all(-1e-6 <= v <= 1.0 + 1e-6 for v in list(road_most.values()) + list(road_least.values()))))
    results.append(("road_score p0 matches deletion_curve's p0 (same image, same target class)", abs(road_most_p0 - del_p0) < 1e-6 and abs(road_least_p0 - del_p0) < 1e-6))

    # 5 (soft check): a real (correctly-localized) CAM should be more "faithful" —
    # lower deletion AUC — than a random CAM, on a deterministic toy model where
    # ground-truth pixel importance is known by construction. This is a sanity
    # check on the metric itself, not a hard mechanical requirement, so a failure
    # is reported as a warning rather than a hard failure.
    real_del_auc = auc(*deletion_curve(model, image, real_quadrant_cam(), device, steps=STEPS, baseline="mean")[:2])
    random_aucs = [
        auc(*deletion_curve(model, image, random_cam(seed=s), device, steps=STEPS, baseline="mean")[:2])
        for s in range(5)
    ]
    mean_random_auc = float(np.mean(random_aucs))
    metric_discriminates = real_del_auc < mean_random_auc
    print(
        f"\n[soft check] real-CAM deletion AUC = {real_del_auc:.4f} vs "
        f"mean random-CAM deletion AUC = {mean_random_auc:.4f} "
        f"({'lower, as expected' if metric_discriminates else 'WARNING: not lower than random'})"
    )

    # 6: normalized curves start at exactly 1.0 (deletion: fraction=0 is the
    # unperturbed image, so probs[0] == p0 by construction, and probs[0]/p0 == 1.0).
    del_probs_norm = del_probs / max(del_p0, 1e-8)
    results.append(("normalized deletion curve starts at exactly 1.0", abs(del_probs_norm[0] - 1.0) < 1e-6))

    # 7: confidence-scale invariance — two models sharing the same underlying
    # importance function but different overall confidence (k) must produce
    # IDENTICAL normalized deletion AUCs, even though their RAW AUCs differ.
    model_hi = ScaledQuadrantModel(k=1.8)
    model_lo = ScaledQuadrantModel(k=1.2)
    model_hi.eval()
    model_lo.eval()

    frac_hi, probs_hi, p0_hi = deletion_curve(model_hi, image, cam, device, steps=STEPS, baseline="mean")
    frac_lo, probs_lo, p0_lo = deletion_curve(model_lo, image, cam, device, steps=STEPS, baseline="mean")

    norm_hi = probs_hi / max(p0_hi, 1e-8)
    norm_lo = probs_lo / max(p0_lo, 1e-8)
    raw_auc_hi = auc(frac_hi, probs_hi)
    raw_auc_lo = auc(frac_lo, probs_lo)
    norm_auc_hi = auc(frac_hi, norm_hi)
    norm_auc_lo = auc(frac_lo, norm_lo)

    results.append(("confidence-scaled models have different p0", abs(p0_hi - p0_lo) > 1e-3))
    results.append(("confidence-scaled models have DIFFERENT raw deletion AUCs (evidence of the un-normalized artifact)", abs(raw_auc_hi - raw_auc_lo) > 1e-3))
    results.append(("confidence-scaled models have IDENTICAL normalized deletion AUCs", abs(norm_auc_hi - norm_auc_lo) < 1e-6))
    results.append(("confidence-scaled models have IDENTICAL normalized curves pointwise", np.allclose(norm_hi, norm_lo, atol=1e-5)))

    # 8: strict checkpoint loading raises on a deliberately mismatched state dict
    model_vanilla = build_mobilenetv3_small(variant="vanilla", num_classes=10, pretrained=False)
    model_no_se = build_mobilenetv3_small(variant="no_se", num_classes=10, pretrained=False)
    strict_load_raised = False
    try:
        model_no_se.load_state_dict(model_vanilla.state_dict(), strict=True)
    except RuntimeError:
        strict_load_raised = True
    results.append(("strict=True load_state_dict raises RuntimeError on a mismatched (vanilla -> no_se) state dict", strict_load_raised))

    # 9: compare_models_statistically on synthetic distributions
    rng = np.random.default_rng(42)
    n = 40
    vals_a = rng.normal(loc=0.5, scale=0.05, size=n)
    vals_b = vals_a + 0.3  # large, consistent, known difference

    diff_comparison = compare_models_statistically({"model_a": vals_a, "model_b": vals_b}, metric_name="toy")
    diff_row = diff_comparison[0]
    results.append(("compare_models_statistically detects a known large difference: t_p < 0.05", diff_row["t_p"] < 0.05))
    results.append(("compare_models_statistically detects a known large difference: wilcoxon_p < 0.05", diff_row["wilcoxon_p"] < 0.05))
    results.append(("compare_models_statistically Cohen's d is large and correctly signed", diff_row["cohens_d"] < -3.0))
    results.append(("compare_models_statistically labels a huge effect as 'large'", diff_row["effect_size"] == "large"))
    results.append(("compare_models_statistically Bonferroni p == raw p for a single comparison (n_comparisons=1)", abs(diff_row["t_p_bonferroni"] - diff_row["t_p"]) < 1e-9))

    same_comparison = compare_models_statistically({"model_a": vals_a, "model_b": vals_a}, metric_name="toy")
    same_row = same_comparison[0]
    results.append(("compare_models_statistically on identical inputs returns t_p ~ 1.0", abs(same_row["t_p"] - 1.0) < 1e-6))
    results.append(("compare_models_statistically on identical inputs returns wilcoxon_p ~ 1.0", abs(same_row["wilcoxon_p"] - 1.0) < 1e-6))
    results.append(("compare_models_statistically on identical inputs returns Cohen's d == 0", same_row["cohens_d"] == 0.0))
    results.append(("compare_models_statistically on identical inputs labels effect as 'negligible'", same_row["effect_size"] == "negligible"))

    # 10: three-way comparison is sorted by |Cohen's d| descending, and Bonferroni
    # correction multiplies by the number of pairwise comparisons (3, for 3 models).
    vals_c = vals_a + 0.02  # small, near-negligible difference from vals_a
    three_way = compare_models_statistically({"model_a": vals_a, "model_b": vals_b, "model_c": vals_c}, metric_name="toy")
    abs_ds = [abs(r["cohens_d"]) for r in three_way]
    results.append(("three-way comparison has 3 pairs (n_comparisons=3)", len(three_way) == 3 and all(r["n_comparisons"] == 3 for r in three_way)))
    results.append(("three-way comparison is sorted by |Cohen's d| descending", abs_ds == sorted(abs_ds, reverse=True)))
    small_pair = next(r for r in three_way if {"model_a", "model_c"} == {r["model_a"], r["model_b"]})
    results.append(("Bonferroni-corrected p >= raw p (n_comparisons=3)", small_pair["t_p_bonferroni"] >= small_pair["t_p"] - 1e-12))

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
