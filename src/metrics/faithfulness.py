"""Quantitative Grad-CAM faithfulness metrics: deletion/insertion AUC, ROAD, and
paired statistical significance testing across model variants.

All curve/score functions take an already-computed CAM (aligned to the model's
input resolution) so the expensive CAM computation itself is never repeated.
Images are expected in the model's normalized input space (see src.data), so
the "mean" baseline — replacing a pixel with the per-channel dataset mean — is
simply zero in that space.

Every curve/score is reported in two forms: RAW (the softmax probability of the
explained class) and confidence-NORMALIZED (the raw curve divided by p0, the
model's initial probability of that class). Raw AUCs are not comparable across
models with different baseline confidence — a model that is simply more
confident everywhere will show a higher raw insertion AUC and a higher raw
deletion AUC without its explanations being any more or less faithful.
Normalizing by p0 measures the FRACTION of a model's own confidence the
explanation accounts for, which is what's actually comparable across models.
The normalized quantities are the headline values throughout this module.
"""

from itertools import combinations
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy import stats
from tqdm import tqdm

from src.data import IMAGENET_MEAN, IMAGENET_STD

CamLike = Union[torch.Tensor, np.ndarray]

BASELINES = ("mean", "black", "blur")

# Cohen's d effect-size bands (Cohen, 1988).
_EFFECT_BANDS = (
    (0.2, "negligible"),
    (0.5, "small"),
    (0.8, "medium"),
)


def effect_size_label(d: float) -> str:
    """Cohen's d magnitude label: negligible (<0.2), small (<0.5), medium (<0.8), large (>=0.8)."""
    ad = abs(d)
    for threshold, label in _EFFECT_BANDS:
        if ad < threshold:
            return label
    return "large"


def _baseline_tensor(image: torch.Tensor, baseline: str) -> torch.Tensor:
    """A full-image baseline tensor in the same (normalized) space as `image`."""
    if baseline == "mean":
        return torch.zeros_like(image)
    if baseline == "black":
        mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(-1, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(-1, 1, 1)
        black_norm = (0.0 - mean) / std
        return black_norm.expand_as(image).clone()
    if baseline == "blur":
        h, w = image.shape[-2:]
        k = min(51, h - (1 - h % 2), w - (1 - w % 2))
        if k % 2 == 0:
            k -= 1
        k = max(k, 3)
        return TF.gaussian_blur(image.unsqueeze(0), kernel_size=[k, k], sigma=[15.0, 15.0])[0]
    raise ValueError(f"Unknown baseline '{baseline}'. Expected one of {BASELINES}.")


def _perturbation_curve(
    model: torch.nn.Module,
    image: torch.Tensor,
    cam: CamLike,
    device: torch.device,
    steps: int,
    baseline: str,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    image = image.to(device)
    c, h, w = image.shape
    n_pixels = h * w

    cam_t = torch.as_tensor(cam, dtype=torch.float32, device=device).reshape(-1)
    order_idx = torch.argsort(cam_t, descending=True)

    baseline_img = _baseline_tensor(image, baseline)

    with torch.no_grad():
        orig_logits = model(image.unsqueeze(0))
        orig_probs = F.softmax(orig_logits, dim=1)[0]
        target_class = int(orig_probs.argmax())
        p0 = float(orig_probs[target_class])

    fractions = np.linspace(0.0, 1.0, steps + 1)
    image_flat = image.reshape(c, n_pixels)
    baseline_flat = baseline_img.reshape(c, n_pixels)

    batch = torch.empty((steps + 1, c, n_pixels), dtype=image.dtype, device=device)
    for i, frac in enumerate(fractions):
        k = int(round(frac * n_pixels))
        if mode == "deletion":
            row = image_flat.clone()
            if k > 0:
                idx = order_idx[:k]
                row[:, idx] = baseline_flat[:, idx]
        else:  # insertion
            row = baseline_flat.clone()
            if k > 0:
                idx = order_idx[:k]
                row[:, idx] = image_flat[:, idx]
        batch[i] = row

    batch = batch.view(steps + 1, c, h, w)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[:, target_class]

    return fractions, probs.detach().cpu().numpy(), p0


def deletion_curve(
    model: torch.nn.Module,
    image: torch.Tensor,
    cam: CamLike,
    device: torch.device,
    steps: int = 20,
    baseline: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Progressively replace the highest-CAM pixels with `baseline`; track
    P(original predicted class). A faithful CAM makes this drop fast (low AUC).
    Returns (fractions, probs, p0) — p0 is the unperturbed probability of the
    explained class, i.e. probs[0]; divide probs by p0 for the normalized curve."""
    return _perturbation_curve(model, image, cam, device, steps, baseline, mode="deletion")


def insertion_curve(
    model: torch.nn.Module,
    image: torch.Tensor,
    cam: CamLike,
    device: torch.device,
    steps: int = 20,
    baseline: str = "blur",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Start fully degraded (`baseline`) and progressively reveal the highest-CAM
    pixels; track P(original predicted class). A faithful CAM makes this rise fast
    (high AUC). Returns (fractions, probs, p0) — p0 is the SAME unperturbed
    probability of the explained class as deletion_curve's (probs[-1] here
    converges to it, since revealing every pixel reconstructs the original image);
    divide probs by p0 for the normalized curve."""
    return _perturbation_curve(model, image, cam, device, steps, baseline, mode="insertion")


def auc(x: Sequence[float], y: Sequence[float]) -> float:
    """Area under (x, y) normalized by the x-range, so AUC is comparable across
    curves that don't span exactly [0, 1] due to rounding."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    x_range = x_arr.max() - x_arr.min()
    if x_range <= 0:
        return float(y_arr.mean())
    return float(np.trapz(y_arr, x_arr) / x_range)


def _road_impute(image: torch.Tensor, remove_mask: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    """ROAD's noisy-linear imputation: each removed pixel is repeatedly replaced
    by the average of its non-removed 8-neighbors, letting known values spread
    inward over a few iterations."""
    c, h, w = image.shape
    device = image.device
    known = (~remove_mask).to(image.dtype)  # (H, W)
    img = image * known.unsqueeze(0)

    neighbor_kernel = torch.ones(1, 1, 3, 3, device=device, dtype=image.dtype)
    neighbor_kernel[0, 0, 1, 1] = 0.0
    channel_kernel = neighbor_kernel.expand(c, 1, 3, 3)

    for _ in range(iterations):
        neighbor_sum = F.conv2d(img.unsqueeze(0), channel_kernel, padding=1, groups=c)[0]
        neighbor_count = F.conv2d(known.view(1, 1, h, w), neighbor_kernel, padding=1)[0, 0]
        has_neighbors = neighbor_count > 0
        avg = neighbor_sum / neighbor_count.clamp_min(1e-6)

        fill_mask = (known == 0) & has_neighbors
        img = torch.where(fill_mask.unsqueeze(0), avg, img)
        known = torch.where(fill_mask, torch.ones_like(known), known)

    return img


def road_score(
    model: torch.nn.Module,
    image: torch.Tensor,
    cam: CamLike,
    device: torch.device,
    percentiles: Sequence[int] = (10, 20, 30, 40, 50),
    order: str = "most",
    iterations: int = 3,
) -> Tuple[Dict[int, float], float]:
    """ROAD (Remove And Debias, Rong et al. 2022): remove the top ('most') or
    bottom ('least') CAM pixels at each percentile and impute them via local
    neighborhood averaging instead of a constant baseline, avoiding deletion's
    out-of-distribution artifact. Returns ({percentile: P(original class)}, p0)."""
    if order not in ("most", "least"):
        raise ValueError(f"order must be 'most' or 'least', got {order!r}")

    model.eval()
    image = image.to(device)
    c, h, w = image.shape
    n_pixels = h * w

    cam_t = torch.as_tensor(cam, dtype=torch.float32, device=device).reshape(-1)
    order_idx = torch.argsort(cam_t, descending=(order == "most"))

    with torch.no_grad():
        orig_logits = model(image.unsqueeze(0))
        orig_probs = F.softmax(orig_logits, dim=1)[0]
        target_class = int(orig_probs.argmax())
        p0 = float(orig_probs[target_class])

    imputed_batch = []
    for pct in percentiles:
        k = int(round(pct / 100.0 * n_pixels))
        remove_flat = torch.zeros(n_pixels, dtype=torch.bool, device=device)
        if k > 0:
            remove_flat[order_idx[:k]] = True
        remove_mask = remove_flat.view(h, w)
        imputed_batch.append(_road_impute(image, remove_mask, iterations=iterations))

    batch = torch.stack(imputed_batch, dim=0)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[:, target_class]

    return {int(pct): float(p) for pct, p in zip(percentiles, probs.detach().cpu().numpy())}, p0


def evaluate_model_faithfulness(
    model: torch.nn.Module,
    dataset,
    indices: Sequence[int],
    device: torch.device,
    cam_fn: Callable[[torch.nn.Module, torch.Tensor, torch.device], CamLike],
    steps: int = 20,
    road_percentiles: Sequence[int] = (10, 20, 30, 40, 50),
    road_iterations: int = 3,
    deletion_baseline: str = "mean",
    insertion_baseline: str = "blur",
    desc: str = "faithfulness",
) -> Tuple[List[Dict], Dict]:
    """Compute deletion/insertion/ROAD faithfulness metrics for one model over a
    fixed set of dataset indices. Returns (per_image_records, aggregate).

    Every metric is recorded both raw and confidence-normalized (divided by p0,
    the model's own initial probability of the explained class); the headline
    fields ("deletion_auc", "insertion_auc", "road_gap") are the NORMALIZED
    values so models with different baseline confidence stay comparable. The
    "_raw" suffixed fields hold the un-normalized values, and "p0" is recorded
    per image so the normalization is auditable."""
    model.eval()
    records: List[Dict] = []

    for idx in tqdm(indices, desc=desc):
        image, label = dataset[idx]
        image = image.to(device)

        cam = cam_fn(model, image, device)

        with torch.no_grad():
            probs0 = F.softmax(model(image.unsqueeze(0)), dim=1)[0]
            pred_class = int(probs0.argmax())

        del_frac, del_probs, p0 = deletion_curve(model, image, cam, device, steps=steps, baseline=deletion_baseline)
        ins_frac, ins_probs, _ = insertion_curve(model, image, cam, device, steps=steps, baseline=insertion_baseline)
        p0_safe = max(p0, 1e-8)

        del_probs_norm = del_probs / p0_safe
        ins_probs_norm = ins_probs / p0_safe

        del_auc_raw = auc(del_frac, del_probs)
        ins_auc_raw = auc(ins_frac, ins_probs)
        del_auc_norm = auc(del_frac, del_probs_norm)
        ins_auc_norm = auc(ins_frac, ins_probs_norm)

        road_most, _ = road_score(model, image, cam, device, percentiles=road_percentiles, order="most", iterations=road_iterations)
        road_least, _ = road_score(model, image, cam, device, percentiles=road_percentiles, order="least", iterations=road_iterations)
        road_most_norm = {k: v / p0_safe for k, v in road_most.items()}
        road_least_norm = {k: v / p0_safe for k, v in road_least.items()}

        road_gap_raw = float(np.mean(list(road_least.values())) - np.mean(list(road_most.values())))
        road_gap_norm = float(np.mean(list(road_least_norm.values())) - np.mean(list(road_most_norm.values())))

        records.append(
            {
                "index": int(idx),
                "true_label": int(label),
                "pred_class": pred_class,
                "correct": pred_class == int(label),
                "p0": p0,
                "deletion_fractions": del_frac.tolist(),
                "deletion_probs_raw": del_probs.tolist(),
                "deletion_probs_norm": del_probs_norm.tolist(),
                "insertion_fractions": ins_frac.tolist(),
                "insertion_probs_raw": ins_probs.tolist(),
                "insertion_probs_norm": ins_probs_norm.tolist(),
                "deletion_auc_raw": del_auc_raw,
                "deletion_auc": del_auc_norm,
                "insertion_auc_raw": ins_auc_raw,
                "insertion_auc": ins_auc_norm,
                "road_most_raw": road_most,
                "road_least_raw": road_least,
                "road_most": road_most_norm,
                "road_least": road_least_norm,
                "road_gap_raw": road_gap_raw,
                "road_gap": road_gap_norm,
            }
        )

    if records:

        def _stat(key: str) -> Tuple[float, float]:
            vals = np.array([r[key] for r in records], dtype=np.float64)
            return float(vals.mean()), float(vals.std())

        del_curves_norm = np.array([r["deletion_probs_norm"] for r in records])
        ins_curves_norm = np.array([r["insertion_probs_norm"] for r in records])
        del_curves_raw = np.array([r["deletion_probs_raw"] for r in records])
        ins_curves_raw = np.array([r["insertion_probs_raw"] for r in records])

        del_auc_mean, del_auc_std = _stat("deletion_auc")
        ins_auc_mean, ins_auc_std = _stat("insertion_auc")
        road_gap_mean, road_gap_std = _stat("road_gap")
        del_auc_raw_mean, del_auc_raw_std = _stat("deletion_auc_raw")
        ins_auc_raw_mean, ins_auc_raw_std = _stat("insertion_auc_raw")
        road_gap_raw_mean, road_gap_raw_std = _stat("road_gap_raw")

        aggregate = {
            "n_images": len(records),
            "accuracy": float(np.mean([r["correct"] for r in records])),
            "mean_p0": float(np.mean([r["p0"] for r in records])),
            "deletion_auc_mean": del_auc_mean,
            "deletion_auc_std": del_auc_std,
            "insertion_auc_mean": ins_auc_mean,
            "insertion_auc_std": ins_auc_std,
            "road_gap_mean": road_gap_mean,
            "road_gap_std": road_gap_std,
            "deletion_auc_raw_mean": del_auc_raw_mean,
            "deletion_auc_raw_std": del_auc_raw_std,
            "insertion_auc_raw_mean": ins_auc_raw_mean,
            "insertion_auc_raw_std": ins_auc_raw_std,
            "road_gap_raw_mean": road_gap_raw_mean,
            "road_gap_raw_std": road_gap_raw_std,
            "fractions": records[0]["deletion_fractions"],
            "mean_deletion_curve": del_curves_norm.mean(axis=0).tolist(),
            "std_deletion_curve": del_curves_norm.std(axis=0).tolist(),
            "mean_insertion_curve": ins_curves_norm.mean(axis=0).tolist(),
            "std_insertion_curve": ins_curves_norm.std(axis=0).tolist(),
            "mean_deletion_curve_raw": del_curves_raw.mean(axis=0).tolist(),
            "std_deletion_curve_raw": del_curves_raw.std(axis=0).tolist(),
            "mean_insertion_curve_raw": ins_curves_raw.mean(axis=0).tolist(),
            "std_insertion_curve_raw": ins_curves_raw.std(axis=0).tolist(),
        }
    else:
        empty_stats = {f"{k}_mean": float("nan") for k in ("deletion_auc", "insertion_auc", "road_gap", "deletion_auc_raw", "insertion_auc_raw", "road_gap_raw")}
        empty_stats.update({f"{k}_std": float("nan") for k in ("deletion_auc", "insertion_auc", "road_gap", "deletion_auc_raw", "insertion_auc_raw", "road_gap_raw")})
        aggregate = {
            "n_images": 0,
            "accuracy": float("nan"),
            "mean_p0": float("nan"),
            **empty_stats,
            "fractions": [],
            "mean_deletion_curve": [],
            "std_deletion_curve": [],
            "mean_insertion_curve": [],
            "std_insertion_curve": [],
            "mean_deletion_curve_raw": [],
            "std_deletion_curve_raw": [],
            "mean_insertion_curve_raw": [],
            "std_insertion_curve_raw": [],
        }

    return records, aggregate


def compare_models_statistically(
    records_by_model: Dict[str, Sequence[float]], metric_name: str = "metric"
) -> List[Dict]:
    """Pairwise paired significance testing (same images, same order, per model).

    For every pair of models: paired t-test, Wilcoxon signed-rank (nonparametric
    fallback), Cohen's d for paired samples (mean diff / std of diffs), a plain-
    language effect-size label, and Bonferroni-corrected p-values (raw p times
    the number of pairwise comparisons for this metric). Results are sorted by
    |Cohen's d| descending — with large n, p-values alone are close to
    uninformative (nearly every pair is "significant"), so effect size is the
    quantity to read first."""
    names = list(records_by_model.keys())
    pairs = list(combinations(names, 2))
    n_comparisons = len(pairs)
    results: List[Dict] = []

    for a, b in pairs:
        vals_a = np.asarray(records_by_model[a], dtype=np.float64)
        vals_b = np.asarray(records_by_model[b], dtype=np.float64)
        if len(vals_a) != len(vals_b):
            raise ValueError(
                f"Paired comparison requires equal-length records for '{a}' ({len(vals_a)}) "
                f"and '{b}' ({len(vals_b)}) computed on the same images in the same order."
            )

        diff = vals_a - vals_b

        if np.allclose(diff, 0.0):
            t_stat, t_p = 0.0, 1.0
            w_p = 1.0
            cohens_d = 0.0
        else:
            t_stat, t_p = stats.ttest_rel(vals_a, vals_b)
            try:
                _, w_p = stats.wilcoxon(vals_a, vals_b)
            except ValueError:
                w_p = float("nan")
            sd = diff.std(ddof=1)
            cohens_d = float(diff.mean() / sd) if sd > 0 else 0.0

        t_p_bonferroni = min(t_p * n_comparisons, 1.0)
        w_p_bonferroni = min(w_p * n_comparisons, 1.0) if np.isfinite(w_p) else float("nan")

        results.append(
            {
                "metric": metric_name,
                "model_a": a,
                "model_b": b,
                "mean_a": float(vals_a.mean()),
                "mean_b": float(vals_b.mean()),
                "diff": float(diff.mean()),
                "t_stat": float(t_stat),
                "t_p": float(t_p),
                "t_p_bonferroni": float(t_p_bonferroni),
                "wilcoxon_p": float(w_p),
                "wilcoxon_p_bonferroni": float(w_p_bonferroni),
                "cohens_d": cohens_d,
                "effect_size": effect_size_label(cohens_d),
                "n_comparisons": n_comparisons,
            }
        )

    results.sort(key=lambda r: abs(r["cohens_d"]), reverse=True)
    return results
