"""Phase 6: quantitative Grad-CAM faithfulness evaluation with paired statistical
significance testing across model variants.

Examples (PowerShell):

    python scripts/faithfulness_eval.py --checkpoints `
        runs/vanilla_scratch/checkpoints/best.pth `
        runs/no_se_scratch/checkpoints/best.pth `
        runs/small_kernel_scratch/checkpoints/best.pth `
        runs/vanilla_finetune/checkpoints/best.pth --num-images 500
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.data import CIFAR10_CLASSES, build_loaders
from src.explain import GradCAM
from src.metrics import compare_models_statistically, evaluate_model_faithfulness
from src.utils import (
    AXIS_COLOR,
    CATEGORICAL_COLORS,
    FIGURE_DPI,
    GRID_COLOR,
    MUTED_TEXT,
    SyntheticTestSet,
    load_model_from_checkpoint,
    resolve_device,
    select_indices,
    set_publication_style,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Grad-CAM faithfulness (deletion/insertion AUC, ROAD) per model and "
        "test cross-model differences for statistical significance."
    )
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths to Phase 2 .pth checkpoints.")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument(
        "--stratified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample equally per class (default: on).",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/faithfulness")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--no-download", action="store_true", help="Use a synthetic random test set instead of CIFAR-10."
    )
    return parser.parse_args()




def make_cam_fn():
    def cam_fn(model, image, device):
        with GradCAM(model) as gradcam:
            cams, _ = gradcam(image.unsqueeze(0).to(device))
        return cams[0].numpy()

    return cam_fn


def plot_curves(aggregates_by_model: dict, colors: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, key, title, ylabel in (
        (axes[0], "deletion", "Deletion curve (lower is more faithful)", "P / P0 (confidence retained)"),
        (axes[1], "insertion", "Insertion curve (higher is more faithful)", "P / P0 (confidence retained)"),
    ):
        for name, agg in aggregates_by_model.items():
            x = np.asarray(agg["fractions"])
            mean = np.asarray(agg[f"mean_{key}_curve"])
            std = np.asarray(agg[f"std_{key}_curve"])
            color = colors[name]
            ax.plot(x, mean, color=color, linewidth=2, label=name)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"Fraction of pixels {'removed' if key == 'deletion' else 'revealed'}")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS_COLOR)
        ax.tick_params(colors=MUTED_TEXT)

    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_auc_bars(aggregates_by_model: dict, colors: dict, output_path: Path) -> None:
    names = list(aggregates_by_model.keys())
    n = len(names)
    x = np.arange(2)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(names):
        agg = aggregates_by_model[name]
        means = [agg["deletion_auc_mean"], agg["insertion_auc_mean"]]
        stds = [agg["deletion_auc_std"], agg["insertion_auc_std"]]
        offset = (i - (n - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width * 0.9,
            yerr=stds,
            capsize=3,
            color=colors[name],
            label=name,
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Deletion AUC (norm)\n(lower = more faithful)", "Insertion AUC (norm)\n(higher = more faithful)"])
    ax.set_ylabel("Normalized AUC (P / P0)")
    ax.set_title("Mean confidence-normalized deletion / insertion AUC per model (+/-1 std)", fontsize=10)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_TEXT)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def print_summary(aggregates_by_model: dict, significance: dict) -> None:
    print("\n=== FAITHFULNESS SUMMARY (confidence-normalized headline metrics, P/P0) ===")
    header = f"{'model':<20}{'acc-on-subset':>15}{'mean P0':>10}{'del AUC norm':>16}{'ins AUC norm':>16}{'ROAD gap norm':>16}"
    print(header)
    print("-" * len(header))
    for name, agg in aggregates_by_model.items():
        print(
            f"{name:<20}{agg['accuracy']:>15.1%}{agg['mean_p0']:>10.3f}"
            f"{agg['deletion_auc_mean']:>11.4f}+/-{agg['deletion_auc_std']:.3f}"
            f"{agg['insertion_auc_mean']:>11.4f}+/-{agg['insertion_auc_std']:.3f}"
            f"{agg['road_gap_mean']:>11.4f}+/-{agg['road_gap_std']:.3f}"
        )
    print(
        "\n(raw, non-normalized AUCs are also saved in faithfulness_metrics.json as "
        "*_auc_raw_mean/std, for reference — they are not directly comparable across "
        "models with different baseline confidence.)"
    )

    print(
        "\n=== PAIRWISE COMPARISONS, sorted by |Cohen's d| (effect size) descending ==="
        "\n(with n=500, p-values alone are close to uninformative — nearly every pair is "
        "\"significant\"; effect size is the quantity to read first. Bonferroni-corrected p "
        "multiplies raw p by the number of comparisons for that metric.)"
    )
    for metric_name, rows in significance.items():
        print(f"\n[{metric_name}]")
        for row in rows:
            print(
                f"  {row['model_a']} vs {row['model_b']}: d={row['cohens_d']:+.3f} ({row['effect_size']}) "
                f"diff={row['diff']:+.4f} (mean_a={row['mean_a']:.4f} mean_b={row['mean_b']:.4f}) "
                f"t_p={row['t_p']:.3g} (bonf={row['t_p_bonferroni']:.3g}) "
                f"wilcoxon_p={row['wilcoxon_p']:.3g} (bonf={row['wilcoxon_p_bonferroni']:.3g})"
            )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    set_publication_style()
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_download:
        base_dataset = SyntheticTestSet(n=max(512, args.num_images), num_classes=len(CIFAR10_CLASSES), seed=args.seed)
    else:
        _, test_loader = build_loaders(root=args.data_root, num_workers=0, download=True)
        base_dataset = test_loader.dataset

    indices = select_indices(base_dataset, args.num_images, args.seed, args.stratified)
    print(f"Selected {len(indices)} shared test images (stratified={args.stratified}).")

    cam_fn = make_cam_fn()

    records_by_model = {}
    aggregates_by_model = {}
    colors = {}

    for i, ckpt in enumerate(args.checkpoints):
        model, name = load_model_from_checkpoint(Path(ckpt), device, check_provenance=True)
        colors[name] = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]

        records, aggregate = evaluate_model_faithfulness(
            model,
            base_dataset,
            indices,
            device,
            cam_fn,
            steps=args.steps,
            desc=f"faithfulness[{name}]",
        )
        records_by_model[name] = records
        aggregates_by_model[name] = aggregate

    metrics_out = {
        name: {"aggregate": aggregates_by_model[name], "records": records_by_model[name]}
        for name in aggregates_by_model
    }
    metrics_path = output_dir / "faithfulness_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved per-model faithfulness metrics to {metrics_path.resolve()}")

    significance = {}
    for metric_name, record_key in (
        ("deletion_auc", "deletion_auc"),
        ("insertion_auc", "insertion_auc"),
        ("road_gap", "road_gap"),
    ):
        values_by_model = {
            name: [r[record_key] for r in records_by_model[name]] for name in records_by_model
        }
        significance[metric_name] = compare_models_statistically(values_by_model, metric_name=metric_name)

    significance_path = output_dir / "significance_tests.json"
    with open(significance_path, "w") as f:
        json.dump(significance, f, indent=2)
    print(f"Saved pairwise significance tests to {significance_path.resolve()}")

    curves_path = output_dir / "deletion_insertion_curves.png"
    plot_curves(aggregates_by_model, colors, curves_path)
    print(f"Saved deletion/insertion curve figure to {curves_path.resolve()}")

    bars_path = output_dir / "auc_bar_chart.png"
    plot_auc_bars(aggregates_by_model, colors, bars_path)
    print(f"Saved AUC bar chart to {bars_path.resolve()}")

    print_summary(aggregates_by_model, significance)


if __name__ == "__main__":
    main()
