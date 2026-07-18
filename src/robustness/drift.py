"""Phase 7: explanation-drift metrics under distribution shift.

For a fixed image, `explanation_drift` compares the Grad-CAM computed on the
clean input against the Grad-CAM computed on a corrupted version of the same
input. `evaluate_robustness` drives this over a dataset, a shared set of
indices, and a grid of (corruption, severity) settings, recording per-image
drift plus whether the corruption flipped the model's prediction.

IMPORTANT (Phase 7.3 fix): drift is measured for a FIXED target class -- the
class the model predicted on the CLEAN image -- so that the clean and
corrupted CAMs are directly comparable (both explain "where is the evidence
for class C"). Explaining the corrupted model's own (possibly different)
prediction instead would conflate the explanation actually moving with the
model simply answering a different question, and inflates drift exactly for
the images/models whose predictions change most under corruption.
"""

import math
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data import IMAGENET_MEAN, IMAGENET_STD, denormalize
from src.explain.gradcam import GradCAM
from src.explain.sanity import spearman_similarity, ssim_similarity

from .corruptions import apply_corruption

TOP_K_FRACTION = 0.20


def _top_k_mask(cam: np.ndarray, frac: float = TOP_K_FRACTION) -> np.ndarray:
    flat = cam.ravel()
    k = max(1, int(round(frac * flat.size)))
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(flat.shape, dtype=bool)
    mask[idx] = True
    return mask.reshape(cam.shape)


def _centroid(cam: np.ndarray) -> np.ndarray:
    total = float(cam.sum())
    h, w = cam.shape
    if total <= 1e-8:
        return np.array([h / 2.0, w / 2.0])
    ys, xs = np.indices(cam.shape)
    cy = float((cam * ys).sum() / total)
    cx = float((cam * xs).sum() / total)
    return np.array([cy, cx])


def explanation_drift(cam_clean: np.ndarray, cam_corrupt: np.ndarray) -> Dict[str, float]:
    """Compare two same-shape CAMs. Perfect agreement (identical CAMs) gives
    spearman=1.0, ssim~1.0, top_k_iou=1.0, centroid_shift=0.0.

    Both CAMs must explain the SAME target class (see `_cam_for_class` below)
    -- otherwise "drift" conflates the explanation moving with the model
    simply answering a different question on the corrupted input."""
    cam_clean = np.asarray(cam_clean, dtype=np.float64)
    cam_corrupt = np.asarray(cam_corrupt, dtype=np.float64)
    if cam_clean.shape != cam_corrupt.shape:
        raise ValueError(f"CAM shape mismatch: {cam_clean.shape} vs {cam_corrupt.shape}")

    spearman = spearman_similarity(cam_clean, cam_corrupt)
    ssim = ssim_similarity(cam_clean, cam_corrupt)

    mask_clean = _top_k_mask(cam_clean)
    mask_corrupt = _top_k_mask(cam_corrupt)
    union = np.logical_or(mask_clean, mask_corrupt).sum()
    intersection = np.logical_and(mask_clean, mask_corrupt).sum()
    top_k_iou = float(intersection / union) if union > 0 else 1.0

    h, w = cam_clean.shape
    diag = math.sqrt(h * h + w * w)
    centroid_shift = float(np.linalg.norm(_centroid(cam_clean) - _centroid(cam_corrupt)) / diag)

    if not (-1.0 - 1e-6 <= spearman <= 1.0 + 1e-6):
        raise ValueError(f"spearman similarity out of [-1, 1] range: {spearman}")
    if not (0.0 - 1e-6 <= top_k_iou <= 1.0 + 1e-6):
        raise ValueError(f"top_k_iou out of [0, 1] range: {top_k_iou}")

    return {
        "spearman": spearman,
        "ssim": ssim,
        "top_k_iou": top_k_iou,
        "centroid_shift": centroid_shift,
    }


def _to_uint8_hwc(image_normalized: torch.Tensor) -> np.ndarray:
    """(C,H,W) normalized tensor -> (H,W,C) uint8 RGB image."""
    img = denormalize(image_normalized.cpu()).permute(1, 2, 0).numpy()
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _to_normalized_tensor(image_uint8_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H,W,C) uint8 RGB image -> (C,H,W) normalized float tensor on `device`."""
    img = image_uint8_hwc.astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return tensor.to(device)


def _cam_for_class(
    model: torch.nn.Module, image: torch.Tensor, device: torch.device, target_class: int = None
) -> Tuple[np.ndarray, int, float, float]:
    """Compute a Grad-CAM plus the model's OWN prediction/confidence on `image`.

    If `target_class` is None, the CAM explains the model's own prediction
    (the clean pass). If `target_class` is given, the CAM explains that FIXED
    class regardless of what the model currently predicts on this image (the
    corrupted pass) -- so drift measures how the explanation for the SAME
    class moved, rather than conflating explanation movement with the model
    answering a different question. `GradCAM.__call__` always returns the
    model's actual argmax prediction (`preds`) independent of `target_class`,
    which is what it is used for here.

    Returns (cam, actual_pred, actual_pred_confidence, explained_class_confidence)
    where explained_class_confidence is the probability of whichever class the
    CAM explains (== actual_pred_confidence when target_class is None).
    """
    with GradCAM(model) as gradcam:
        cams, preds = gradcam(image.unsqueeze(0).to(device), target_class=target_class)
    pred = int(preds[0])
    explained_class = int(target_class) if target_class is not None else pred
    with torch.no_grad():
        probs = F.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0]
    confidence = float(probs[pred])
    explained_confidence = float(probs[explained_class])
    return cams[0].numpy(), pred, confidence, explained_confidence


def evaluate_robustness(
    model: torch.nn.Module,
    dataset,
    indices: Sequence[int],
    device: torch.device,
    corruptions: Sequence[str],
    severities: Sequence[int],
    desc: str = "robustness",
) -> Tuple[List[Dict], Dict]:
    """For every image: compute the clean CAM/prediction once, then for every
    (corruption, severity) corrupt the un-normalized image and recompute the
    CAM -- but for the FIXED clean-predicted class, not whatever the model
    predicts on the corrupted input. This keeps clean and corrupted CAMs
    answering the same question ("where does the evidence for class C live?")
    so that `explanation_drift` measures the explanation actually moving,
    rather than conflating that with the model's prediction changing (which is
    tracked separately via `corrupt_pred` / `flipped`). Returns
    (per_image_records, aggregates_by_corruption_severity)."""
    model.eval()
    records: List[Dict] = []

    total = len(indices) * len(corruptions) * len(severities)
    pbar = tqdm(total=total, desc=desc)

    for idx in indices:
        image, label = dataset[idx]
        image = image.to(device)
        label = int(label)

        cam_clean, pred_clean, conf_clean, _ = _cam_for_class(model, image, device, target_class=None)
        clean_correct = pred_clean == label
        image_uint8 = _to_uint8_hwc(image)

        for corruption in corruptions:
            for severity in severities:
                corrupt_uint8 = apply_corruption(image_uint8, corruption, severity)
                corrupt_tensor = _to_normalized_tensor(corrupt_uint8, device)

                cam_corrupt, pred_corrupt, conf_corrupt, conf_corrupt_of_clean_class = _cam_for_class(
                    model, corrupt_tensor, device, target_class=pred_clean
                )
                drift = explanation_drift(cam_clean, cam_corrupt)
                corrupt_correct = pred_corrupt == label
                flipped = pred_corrupt != pred_clean

                records.append(
                    {
                        "index": idx,
                        "true_label": label,
                        "corruption": corruption,
                        "severity": severity,
                        "clean_pred": pred_clean,
                        "clean_confidence": conf_clean,
                        "clean_correct": clean_correct,
                        "corrupt_pred": pred_corrupt,
                        "corrupt_confidence": conf_corrupt,
                        "corrupt_correct": corrupt_correct,
                        "conf_corrupt_of_clean_class": conf_corrupt_of_clean_class,
                        "explained_class": pred_clean,
                        "flipped": flipped,
                        **drift,
                    }
                )
                pbar.update(1)

    pbar.close()

    aggregates = _aggregate(records)
    return records, aggregates


def _aggregate(records: List[Dict]) -> Dict[str, Dict]:
    groups: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for r in records:
        groups[(r["corruption"], r["severity"])].append(r)

    aggregates: Dict[str, Dict] = {}
    for (corruption, severity), rows in groups.items():
        key = f"{corruption}|sev{severity}"
        n = len(rows)

        def _mean(field: str, subset: List[Dict] = rows) -> float:
            return float(np.mean([r[field] for r in subset])) if subset else float("nan")

        stable = [r for r in rows if not r["flipped"]]
        flipped = [r for r in rows if r["flipped"]]

        aggregates[key] = {
            "corruption": corruption,
            "severity": severity,
            "n": n,
            "mean_spearman": _mean("spearman"),
            "mean_ssim": _mean("ssim"),
            "mean_top_k_iou": _mean("top_k_iou"),
            "mean_centroid_shift": _mean("centroid_shift"),
            "accuracy_under_corruption": _mean("corrupt_correct"),
            "flip_rate": _mean("flipped"),
            "n_stable": len(stable),
            "n_flipped": len(flipped),
            "mean_spearman_stable": _mean("spearman", stable) if stable else float("nan"),
            "mean_spearman_flipped": _mean("spearman", flipped) if flipped else float("nan"),
            "mean_top_k_iou_stable": _mean("top_k_iou", stable) if stable else float("nan"),
            "mean_top_k_iou_flipped": _mean("top_k_iou", flipped) if flipped else float("nan"),
            "mean_centroid_shift_stable": _mean("centroid_shift", stable) if stable else float("nan"),
            "mean_centroid_shift_flipped": _mean("centroid_shift", flipped) if flipped else float("nan"),
        }

    return aggregates


DRIFT_METRICS: Tuple[str, ...] = ("spearman", "ssim", "top_k_iou", "centroid_shift")

# Higher spearman/ssim/top_k_iou means LESS drift (more stable); higher
# centroid_shift means MORE drift. Flip sign so "mean drift" is uniformly
# "bigger number = more explanation instability" across all four metrics.
HIGHER_IS_MORE_DRIFT: Dict[str, bool] = {
    "spearman": False,
    "ssim": False,
    "top_k_iou": False,
    "centroid_shift": True,
}


def drift_score(row: Dict, metric: str) -> float:
    """A single number where BIGGER == MORE drift, for every metric in DRIFT_METRICS."""
    return (1.0 - row[metric]) if not HIGHER_IS_MORE_DRIFT[metric] else row[metric]

    return aggregates
