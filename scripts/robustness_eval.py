"""Phase 7: explanation robustness under distribution shift.

Question: when inputs are corrupted, do explanations DRIFT -- and do
architectural variants differ in explanation stability even where they don't
differ in static faithfulness (Phase 6.1 found negligible effects there)?

For a shared, seeded set of test images (paired design across models), each
model's clean Grad-CAM is compared against its Grad-CAM on corrupted versions
of the same image, across a grid of corruption types and severities.

Example (PowerShell):

    python scripts/robustness_eval.py `
        --checkpoints runs/vanilla_scratch/checkpoints/best.pth `
        runs/no_se_scratch/checkpoints/best.pth `
        runs/small_kernel_scratch/checkpoints/best.pth `
        runs/vanilla_finetune/checkpoints/best.pth --num-images 200
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.data import CIFAR10_CLASSES, build_loaders
from src.metrics import effect_size_label
from src.robustness import CORRUPTIONS, DRIFT_METRICS, drift_score, evaluate_robustness
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths to Phase 2 .pth checkpoints.")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument(
        "--stratified", action=argparse.BooleanOptionalAction, default=True, help="Sample equally per class (default: on)."
    )
    parser.add_argument("--corruptions", nargs="+", default=list(CORRUPTIONS), choices=list(CORRUPTIONS))
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--output-dir", default="runs/robustness")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true", help="Use a synthetic random test set instead of CIFAR-10.")
    return parser.parse_args()


def compute_significance(records_by_model: dict) -> dict:
    """Pairwise paired comparisons on MEAN DRIFT PER IMAGE (averaged over every
    corruption x severity combination), same images across models -- exactly
    the paired-design pattern from Phase 6.1's compare_models_statistically."""
    names = list(records_by_model.keys())
    pairs = list(combinations(names, 2))

    significance = {}
    for metric in DRIFT_METRICS:
        # per-model: mean drift score per image index (averaged across corruption/severity)
        per_image_by_model = {}
        for name in names:
            by_idx = defaultdict(list)
            for r in records_by_model[name]:
                by_idx[r["index"]].append(drift_score(r, metric))
            ref_order = sorted(by_idx.keys())
            per_image_by_model[name] = (ref_order, np.array([np.mean(by_idx[i]) for i in ref_order]))

        n_comparisons = len(pairs)
        rows = []
        for a, b in pairs:
            idx_a, vals_a = per_image_by_model[a]
            idx_b, vals_b = per_image_by_model[b]
            if idx_a != idx_b:
                raise ValueError(f"Image index mismatch between '{a}' and '{b}'; cannot pair by index.")

            diff = vals_a - vals_b
            if np.allclose(diff, 0.0):
                t_stat, t_p, w_p, cohens_d = 0.0, 1.0, 1.0, 0.0
            else:
                t_stat, t_p = stats.ttest_rel(vals_a, vals_b)
                try:
                    _, w_p = stats.wilcoxon(vals_a, vals_b)
                except ValueError:
                    w_p = float("nan")
                sd = diff.std(ddof=1)
                cohens_d = float(diff.mean() / sd) if sd > 0 else 0.0

            t_p_bonf = min(t_p * n_comparisons, 1.0)
            w_p_bonf = min(w_p * n_comparisons, 1.0) if np.isfinite(w_p) else float("nan")

            rows.append(
                {
                    "metric": f"{metric}_drift",
                    "model_a": a,
                    "model_b": b,
                    "mean_a": float(vals_a.mean()),
                    "mean_b": float(vals_b.mean()),
                    "diff": float(diff.mean()),
                    "t_stat": float(t_stat),
                    "t_p": float(t_p),
                    "t_p_bonferroni": float(t_p_bonf),
                    "wilcoxon_p": float(w_p),
                    "wilcoxon_p_bonferroni": float(w_p_bonf),
                    "cohens_d": cohens_d,
                    "effect_size": effect_size_label(cohens_d),
                    "n_comparisons": n_comparisons,
                }
            )
        rows.sort(key=lambda r: abs(r["cohens_d"]), reverse=True)
        significance[f"{metric}_drift"] = rows

    return significance


def plot_drift_vs_severity(aggregates_by_model: dict, corruptions: list, severities: list, colors: dict, output_path: Path) -> None:
    n_corr = len(corruptions)
    n_cols = min(3, n_corr)
    n_rows = (n_corr + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)

    for i, corruption in enumerate(corruptions):
        ax = axes[i // n_cols][i % n_cols]
        for name, agg in aggregates_by_model.items():
            y = [1.0 - agg[f"{corruption}|sev{s}"]["mean_spearman"] for s in severities]
            ax.plot(severities, y, marker="o", color=colors[name], linewidth=2, label=name)
        ax.set_title(corruption, fontsize=10)
        ax.set_xlabel("severity")
        ax.set_ylabel("mean explanation drift\n(1 - spearman)")
        ax.set_xticks(severities)
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS_COLOR)
        ax.tick_params(colors=MUTED_TEXT)

    for j in range(n_corr, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Mean explanation drift (1 - Spearman) vs corruption severity", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_severity(aggregates_by_model: dict, corruptions: list, severities: list, colors: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, agg in aggregates_by_model.items():
        # mean over corruption types at each severity
        y = [np.mean([agg[f"{c}|sev{s}"]["accuracy_under_corruption"] for c in corruptions]) for s in severities]
        ax.plot(severities, y, marker="o", color=colors[name], linewidth=2, label=name)

    ax.set_xlabel("severity")
    ax.set_ylabel("accuracy (mean over corruption types)")
    ax.set_xticks(severities)
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy under corruption vs severity", fontsize=10)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_TEXT)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def print_summary_table(clean_accuracy: dict, aggregates_by_model: dict, corruptions: list) -> None:
    print("\n=== SUMMARY TABLE ===")
    header = f"{'model':<20}{'clean acc':>11}{'acc@sev3':>11}{'mean spearman drift':>22}{'mean top-k IoU':>16}{'flip rate':>11}"
    print(header)
    print("-" * len(header))
    for name, agg in aggregates_by_model.items():
        sev3_accs = [agg[f"{c}|sev3"]["accuracy_under_corruption"] for c in corruptions if f"{c}|sev3" in agg]
        acc_sev3 = float(np.mean(sev3_accs)) if sev3_accs else float("nan")
        mean_spearman_drift = float(np.mean([1.0 - v["mean_spearman"] for v in agg.values()]))
        mean_iou = float(np.mean([v["mean_top_k_iou"] for v in agg.values()]))
        mean_flip = float(np.mean([v["flip_rate"] for v in agg.values()]))
        print(
            f"{name:<20}{clean_accuracy[name]:>11.1%}{acc_sev3:>11.1%}"
            f"{mean_spearman_drift:>22.4f}{mean_iou:>16.4f}{mean_flip:>11.1%}"
        )


def print_key_analysis(aggregates_by_model: dict, clean_accuracy: dict, corruptions: list, severities: list) -> None:
    print("\n=== KEY ANALYSIS: does explanation drift exceed what accuracy degradation alone predicts? ===")
    print(
        "For each model, we compare mean explanation drift (1 - spearman, averaged over all corruptions/severities) "
        "against mean accuracy DROP (clean acc - acc under corruption). If drift scales roughly with accuracy drop, "
        "explanations are just tracking prediction correctness; if drift is large even when accuracy barely moves, "
        "explanations are unstable independent of whether the prediction is right.\n"
    )
    header = f"{'model':<20}{'mean acc drop':>15}{'mean spearman drift':>22}{'drift / acc-drop ratio':>24}"
    print(header)
    print("-" * len(header))
    for name, agg in aggregates_by_model.items():
        acc_drop = clean_accuracy[name] - float(np.mean([v["accuracy_under_corruption"] for v in agg.values()]))
        mean_drift = float(np.mean([1.0 - v["mean_spearman"] for v in agg.values()]))
        ratio = mean_drift / acc_drop if acc_drop > 1e-6 else float("inf")
        print(f"{name:<20}{acc_drop:>15.4f}{mean_drift:>22.4f}{ratio:>24.2f}")

    print(
        "\n(ratio >> a shared baseline across models indicates that model's explanations move MORE than its "
        "accuracy loss alone would predict -- i.e., unstable explanations even when predictions hold steady.)"
    )

    print("\n--- drift split by whether the prediction flipped (stayed correct vs flipped) ---")
    header2 = f"{'model':<20}{'spearman|stable':>17}{'spearman|flipped':>18}{'IoU|stable':>13}{'IoU|flipped':>13}"
    print(header2)
    print("-" * len(header2))
    for name, agg in aggregates_by_model.items():
        stable_vals = [v["mean_spearman_stable"] for v in agg.values() if not np.isnan(v["mean_spearman_stable"])]
        flipped_vals = [v["mean_spearman_flipped"] for v in agg.values() if not np.isnan(v["mean_spearman_flipped"])]
        iou_stable = [v["mean_top_k_iou_stable"] for v in agg.values() if not np.isnan(v["mean_top_k_iou_stable"])]
        iou_flipped = [v["mean_top_k_iou_flipped"] for v in agg.values() if not np.isnan(v["mean_top_k_iou_flipped"])]
        s_stable = np.mean(stable_vals) if stable_vals else float("nan")
        s_flipped = np.mean(flipped_vals) if flipped_vals else float("nan")
        i_stable = np.mean(iou_stable) if iou_stable else float("nan")
        i_flipped = np.mean(iou_flipped) if iou_flipped else float("nan")
        print(f"{name:<20}{s_stable:>17.4f}{s_flipped:>18.4f}{i_stable:>13.4f}{i_flipped:>13.4f}")
    print(
        "\n(spearman/IoU|stable measures drift on images where the prediction did NOT flip under corruption -- "
        "the interesting case: an explanation that moves substantially even though the model's answer held.)"
    )


def main() -> None:
    args = parse_args()
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
    print(f"Corruptions: {args.corruptions}")
    print(f"Severities: {args.severities}")
    print(
        f"Total CAM evaluations per model: {len(indices)} clean + "
        f"{len(indices) * len(args.corruptions) * len(args.severities)} corrupted "
        f"= {len(indices) * (1 + len(args.corruptions) * len(args.severities))}"
    )

    records_by_model = {}
    aggregates_by_model = {}
    clean_accuracy = {}
    colors = {}

    for i, ckpt in enumerate(args.checkpoints):
        model, name = load_model_from_checkpoint(Path(ckpt), device, check_provenance=True)
        colors[name] = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]

        records, aggregate = evaluate_robustness(
            model, base_dataset, indices, device, args.corruptions, args.severities, desc=f"robustness[{name}]"
        )
        records_by_model[name] = records
        aggregates_by_model[name] = aggregate

        clean_correct = {}
        for r in records:
            clean_correct[r["index"]] = r["clean_correct"]
        clean_accuracy[name] = float(np.mean(list(clean_correct.values())))

    metrics_out = {
        name: {"clean_accuracy": clean_accuracy[name], "aggregate": aggregates_by_model[name], "records": records_by_model[name]}
        for name in aggregates_by_model
    }
    metrics_path = output_dir / "robustness_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved per-model robustness metrics to {metrics_path.resolve()}")

    significance = compute_significance(records_by_model)
    significance_path = output_dir / "significance_tests.json"
    with open(significance_path, "w") as f:
        json.dump(significance, f, indent=2)
    print(f"Saved pairwise significance tests to {significance_path.resolve()}")

    drift_fig_path = output_dir / "drift_vs_severity.png"
    plot_drift_vs_severity(aggregates_by_model, args.corruptions, args.severities, colors, drift_fig_path)
    print(f"Saved drift-vs-severity figure to {drift_fig_path.resolve()}")

    acc_fig_path = output_dir / "accuracy_vs_severity.png"
    plot_accuracy_vs_severity(aggregates_by_model, args.corruptions, args.severities, colors, acc_fig_path)
    print(f"Saved accuracy-vs-severity figure to {acc_fig_path.resolve()}")

    print_summary_table(clean_accuracy, aggregates_by_model, args.corruptions)
    print_key_analysis(aggregates_by_model, clean_accuracy, args.corruptions, args.severities)

    print("\n=== PAIRWISE SIGNIFICANCE TESTS ON MEAN DRIFT (paired, sorted by |Cohen's d| descending) ===")
    for metric_name, rows in significance.items():
        print(f"\n[{metric_name}]")
        for row in rows:
            print(
                f"  {row['model_a']} vs {row['model_b']}: d={row['cohens_d']:+.3f} ({row['effect_size']}) "
                f"diff={row['diff']:+.4f} (mean_a={row['mean_a']:.4f} mean_b={row['mean_b']:.4f}) "
                f"t_p={row['t_p']:.3g} (bonf={row['t_p_bonferroni']:.3g}) "
                f"wilcoxon_p={row['wilcoxon_p']:.3g} (bonf={row['wilcoxon_p_bonferroni']:.3g})"
            )


if __name__ == "__main__":
    main()
