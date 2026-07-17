import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.explain import build_comparison, render_comparison_grid, select_shared_indices
from src.models import build_mobilenetv3_small

NUM_CLASSES = 10
CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]


class SyntheticTestSet(torch.utils.data.Dataset):
    def __init__(self, n: int, num_classes: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, 224, 224, generator=g) * 0.25
        self.labels = torch.randint(0, num_classes, (n,), generator=g).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def main() -> bool:
    results = []
    torch.manual_seed(0)
    device = torch.device("cpu")

    test_set = SyntheticTestSet(n=24, num_classes=NUM_CLASSES, seed=1)

    models_and_names = [
        (build_mobilenetv3_small(variant="vanilla", num_classes=NUM_CLASSES, pretrained=False), "vanilla"),
        (build_mobilenetv3_small(variant="no_se", num_classes=NUM_CLASSES, pretrained=False), "no_se"),
        (build_mobilenetv3_small(variant="small_kernel", num_classes=NUM_CLASSES, pretrained=False), "small_kernel"),
    ]
    for model, _name in models_and_names:
        model.to(device)
        model.eval()

    num_images = 4

    # 1: select_shared_indices — count, validity, and single shared set
    indices = select_shared_indices(test_set, models_and_names, device, num_images=num_images, seed=42)
    results.append(("select_shared_indices returns requested count", len(indices) == num_images))
    results.append(
        ("select_shared_indices returns valid indices", all(0 <= i < len(test_set) for i in indices))
    )
    indices_again = select_shared_indices(test_set, models_and_names, device, num_images=num_images, seed=42)
    results.append(("select_shared_indices is deterministic for a fixed seed (same set every model)", indices == indices_again))

    # 2 & 3: build_comparison structure, both target modes
    for target in ("true", "pred"):
        result = build_comparison(models_and_names, test_set, indices, device, target=target)
        ok_len = len(result) == len(indices)
        ok_per_model = all(len(item["per_model"]) == len(models_and_names) for item in result)
        ok_shapes = all(
            pm["cam"].shape == item["original_image"].shape[:2] for item in result for pm in item["per_model"]
        )
        ok_range = all(
            (pm["cam"].min() >= -1e-6) and (pm["cam"].max() <= 1.0 + 1e-6)
            for item in result
            for pm in item["per_model"]
        )
        results.append((f"build_comparison (target={target}) returns one entry per image", ok_len))
        results.append((f"build_comparison (target={target}) has one cam per model per image", ok_per_model))
        results.append((f"build_comparison (target={target}) cam shape == image HxW", ok_shapes))
        results.append((f"build_comparison (target={target}) cam values in [0,1]", ok_range))

    # 4: render_comparison_grid writes a nonzero-size PNG
    result_true = build_comparison(models_and_names, test_set, indices, device, target="true")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "compare_grid.png"
        render_comparison_grid(result_true, CLASS_NAMES, out_path)
        results.append(("render_comparison_grid writes a PNG", out_path.exists()))
        results.append(("render_comparison_grid PNG is nonzero size", out_path.exists() and out_path.stat().st_size > 0))

    # 5: only_all_correct path runs without error
    try:
        oac_indices = select_shared_indices(
            test_set, models_and_names, device, num_images=num_images, seed=42, only_all_correct=True
        )
        oac_result = build_comparison(models_and_names, test_set, oac_indices, device, target="true")
        oac_ok = len(oac_result) == len(oac_indices) and len(oac_indices) <= num_images
        if oac_indices:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = Path(tmpdir) / "compare_grid_oac.png"
                render_comparison_grid(oac_result, CLASS_NAMES, out_path)
                oac_ok = oac_ok and out_path.exists()
    except Exception as e:  # noqa: BLE001
        oac_ok = False
        print(f"only_all_correct path raised: {e}")
    results.append(("only_all_correct path runs without error", oac_ok))

    print("\n=== COMPARE SMOKE TEST RESULTS ===")
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
