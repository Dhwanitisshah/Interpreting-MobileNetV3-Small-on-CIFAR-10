"""Mechanics-only smoke test for src/robustness -- no download, synthetic data.

Verifies: every corruption returns a valid same-shape uint8 image that actually
changes the input; severity monotonically increases the amount of change;
explanation_drift of a CAM against itself is perfect; drift against a shifted
CAM is strictly worse; evaluate_robustness returns the expected record/
aggregate structure on a tiny synthetic dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.robustness import CORRUPTIONS, apply_corruption, evaluate_robustness, explanation_drift

IMG_SIZE = 224


class TinyModel(nn.Module):
    """Deterministic, parameter-light classifier: fine for exercising the
    Grad-CAM + corruption pipeline without a real trained checkpoint."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)
        torch.manual_seed(0)
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.5)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        return self.fc(pooled)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 6, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n, 3, IMG_SIZE, IMG_SIZE, generator=g)
        self.labels = torch.randint(0, num_classes, (n,), generator=g).tolist()
        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def make_uint8_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)


def make_cam(seed: int, shape=(224, 224)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape).astype(np.float32)


def main() -> bool:
    results = []
    base_image = make_uint8_image(seed=1)

    # 1: every corruption returns a valid uint8 image of the same shape, and
    # actually changes it; severity monotonically increases the amount of change.
    for name in CORRUPTIONS:
        prev_diff = -1.0
        monotonic = True
        for severity in range(1, 6):
            out = apply_corruption(base_image, name, severity)
            valid_shape = out.shape == base_image.shape and out.dtype == np.uint8
            changed = not np.array_equal(out, base_image)
            results.append((f"[{name}] severity={severity} returns valid uint8 image of same shape", valid_shape))
            results.append((f"[{name}] severity={severity} actually changes the image", changed))

            diff = float(np.mean(np.abs(out.astype(np.int32) - base_image.astype(np.int32))))
            if severity > 1 and diff < prev_diff - 1e-6:
                monotonic = False
            prev_diff = diff
        results.append((f"[{name}] mean abs pixel change is (roughly) monotonic non-decreasing in severity", monotonic))

    # 2: drift of a CAM against itself is perfect.
    cam = make_cam(seed=2)
    self_drift = explanation_drift(cam, cam)
    results.append(("explanation_drift(cam, cam) spearman == 1.0", abs(self_drift["spearman"] - 1.0) < 1e-6))
    results.append(("explanation_drift(cam, cam) ssim ~= 1.0", abs(self_drift["ssim"] - 1.0) < 1e-3))
    results.append(("explanation_drift(cam, cam) top_k_iou == 1.0", abs(self_drift["top_k_iou"] - 1.0) < 1e-9))
    results.append(("explanation_drift(cam, cam) centroid_shift == 0.0", abs(self_drift["centroid_shift"]) < 1e-9))

    # 3: drift against a shifted (spatially rolled + independently randomized) CAM
    # is worse than drift against itself, on every axis.
    shifted_cam = np.roll(make_cam(seed=3), shift=(cam.shape[0] // 2, cam.shape[1] // 2), axis=(0, 1))
    shift_drift = explanation_drift(cam, shifted_cam)
    results.append(("shifted-CAM spearman < self spearman", shift_drift["spearman"] < self_drift["spearman"]))
    results.append(("shifted-CAM ssim < self ssim", shift_drift["ssim"] < self_drift["ssim"]))
    results.append(("shifted-CAM top_k_iou < self top_k_iou", shift_drift["top_k_iou"] < self_drift["top_k_iou"]))
    results.append(("shifted-CAM centroid_shift > self centroid_shift", shift_drift["centroid_shift"] > self_drift["centroid_shift"]))

    # 4: evaluate_robustness end-to-end structure on a tiny synthetic dataset.
    device = torch.device("cpu")
    model = TinyModel()
    model.eval()
    dataset = SyntheticDataset(n=4, seed=7)
    indices = list(range(len(dataset)))
    corruptions = ["gaussian_noise", "jpeg_compression"]
    severities = [1, 3]

    records, aggregates = evaluate_robustness(model, dataset, indices, device, corruptions, severities, desc="smoke")

    expected_n_records = len(indices) * len(corruptions) * len(severities)
    results.append(("evaluate_robustness returns one record per (image, corruption, severity)", len(records) == expected_n_records))

    required_keys = {
        "index", "true_label", "corruption", "severity", "clean_pred", "clean_confidence",
        "clean_correct", "corrupt_pred", "corrupt_confidence", "corrupt_correct", "flipped",
        "spearman", "ssim", "top_k_iou", "centroid_shift",
    }
    results.append(("every record has the expected keys", all(required_keys.issubset(r.keys()) for r in records)))
    results.append(
        ("every record's drift metrics are finite and in sane ranges",
         all(
             -1.0 - 1e-6 <= r["spearman"] <= 1.0 + 1e-6
             and 0.0 - 1e-6 <= r["top_k_iou"] <= 1.0 + 1e-6
             and r["centroid_shift"] >= 0.0
             for r in records
         )),
    )

    expected_n_groups = len(corruptions) * len(severities)
    results.append(("aggregates has one entry per (corruption, severity)", len(aggregates) == expected_n_groups))
    agg_keys = {
        "corruption", "severity", "n", "mean_spearman", "mean_ssim", "mean_top_k_iou",
        "mean_centroid_shift", "accuracy_under_corruption", "flip_rate", "n_stable", "n_flipped",
    }
    results.append(("every aggregate entry has the expected keys", all(agg_keys.issubset(v.keys()) for v in aggregates.values())))
    results.append(("every aggregate group has n == num_images", all(v["n"] == len(indices) for v in aggregates.values())))
    results.append(("n_stable + n_flipped == n in every aggregate group", all(v["n_stable"] + v["n_flipped"] == v["n"] for v in aggregates.values())))

    print("\n=== ROBUSTNESS SMOKE TEST RESULTS ===")
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
