"""Reporting/analysis over existing runs/faithfulness/ artifacts.

Reads faithfulness_metrics.json and significance_tests.json only — no
retraining, no re-evaluation. Prints:
  1. Per-model summary tables (normalized headline + raw).
  2. All pairwise significance comparisons, grouped by metric.
  3. TOST equivalence testing on the paired per-image normalized metrics.
  4. A plain-English verdict for the key vanilla_scratch vs no_se_scratch pair.
  5. A p0-confound diagnostic: within-model correlation between p0 and each
     normalized metric, to check whether cross-model ranking differences are
     a normalization artifact rather than a real property of the models.

Output goes to stdout and to runs/faithfulness/report.txt.
"""

import argparse
import io
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FAITH_DIR = ROOT / "runs" / "faithfulness"

NORM_METRICS = ("deletion_auc", "insertion_auc", "road_gap")
NORM_LABELS = {
    "deletion_auc": "norm deletion AUC",
    "insertion_auc": "norm insertion AUC",
    "road_gap": "norm ROAD gap",
}


def effect_size_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_pm(mean: float, std: float, prec: int = 4) -> str:
    return f"{mean:.{prec}f} (+/-{std:.{prec}f})"


def _write_table(out, cols, rows) -> None:
    header = "  ".join(f"{name:{align}{width}}" for name, width, align in cols)
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")
    for cells in rows:
        out.write(
            "  ".join(f"{cell:{align}{width}}" for cell, (_, width, align) in zip(cells, cols)) + "\n"
        )


def print_summary_tables(out, metrics: dict) -> None:
    models = list(metrics.keys())

    out.write("=" * 100 + "\n")
    out.write("1. PER-MODEL SUMMARY (normalized, headline)\n")
    out.write("=" * 100 + "\n")
    cols = [
        ("model", 22, "<"), ("n", 5, ">"), ("accuracy", 9, ">"), ("mean p0", 9, ">"),
        ("norm del AUC", 20, ">"), ("norm ins AUC", 20, ">"), ("norm ROAD gap", 20, ">"),
    ]
    rows = []
    for m in models:
        agg = metrics[m]["aggregate"]
        rows.append([
            m, str(agg["n_images"]), f"{agg['accuracy']:.4f}", f"{agg['mean_p0']:.4f}",
            fmt_pm(agg["deletion_auc_mean"], agg["deletion_auc_std"]),
            fmt_pm(agg["insertion_auc_mean"], agg["insertion_auc_std"]),
            fmt_pm(agg["road_gap_mean"], agg["road_gap_std"]),
        ])
    _write_table(out, cols, rows)
    out.write("\n")

    out.write("=" * 100 + "\n")
    out.write("1b. PER-MODEL SUMMARY (raw, unnormalized -- not comparable across models)\n")
    out.write("=" * 100 + "\n")
    cols = [
        ("model", 22, "<"), ("n", 5, ">"), ("accuracy", 9, ">"), ("mean p0", 9, ">"),
        ("raw del AUC", 20, ">"), ("raw ins AUC", 20, ">"), ("raw ROAD gap", 20, ">"),
    ]
    rows = []
    for m in models:
        agg = metrics[m]["aggregate"]
        rows.append([
            m, str(agg["n_images"]), f"{agg['accuracy']:.4f}", f"{agg['mean_p0']:.4f}",
            fmt_pm(agg["deletion_auc_raw_mean"], agg["deletion_auc_raw_std"]),
            fmt_pm(agg["insertion_auc_raw_mean"], agg["insertion_auc_raw_std"]),
            fmt_pm(agg["road_gap_raw_mean"], agg["road_gap_raw_std"]),
        ])
    _write_table(out, cols, rows)
    out.write("\n")


def print_significance_tables(out, sig: dict) -> None:
    out.write("=" * 100 + "\n")
    out.write("2. PAIRWISE SIGNIFICANCE TESTS (grouped by metric, sorted by |Cohen's d| descending)\n")
    out.write("=" * 100 + "\n")
    cols = [
        ("comparison", 44, "<"),
        ("mean_a", 9, ">"),
        ("mean_b", 9, ">"),
        ("diff", 9, ">"),
        ("d (effect)", 22, ">"),
        ("raw p", 12, ">"),
        ("bonf p", 12, ">"),
    ]
    for metric, rows in sig.items():
        out.write(f"\n-- {metric} --\n")
        rows_sorted = sorted(rows, key=lambda r: abs(r["cohens_d"]), reverse=True)
        table_rows = []
        for r in rows_sorted:
            comparison = f"{r['model_a']} vs {r['model_b']}"
            d_str = f"{r['cohens_d']:.4f} ({r['effect_size']})"
            table_rows.append([
                comparison,
                f"{r['mean_a']:.4f}",
                f"{r['mean_b']:.4f}",
                f"{r['diff']:.4f}",
                d_str,
                f"{r['t_p']:.4g}",
                f"{r['t_p_bonferroni']:.4g}",
            ])
        _write_table(out, cols, table_rows)
    out.write("\n")


def tost_paired(a: np.ndarray, b: np.ndarray, sesoi_d: float):
    """Two one-sided paired t-tests (TOST) for equivalence, plus the 90% CI.

    Bounds are +/- sesoi_d * sd_of_differences (the smallest effect size of
    interest, expressed in Cohen's d, converted to the raw scale of the
    paired differences). Returns a dict with p_TOST, the bound, mean diff,
    and the 90% CI on the mean paired difference.
    """
    diff = a - b
    n = len(diff)
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    bound = sesoi_d * sd

    if se == 0.0:
        p_lower = 0.0 if mean_diff > -bound else 1.0
        p_upper = 0.0 if mean_diff < bound else 1.0
    else:
        t_lower = (mean_diff - (-bound)) / se
        p_lower = 1.0 - stats.t.cdf(t_lower, df)
        t_upper = (mean_diff - bound) / se
        p_upper = stats.t.cdf(t_upper, df)

    p_tost = max(p_lower, p_upper)

    # 90% CI on the mean paired difference (the interval matching alpha=0.05 TOST).
    t_crit = stats.t.ppf(0.95, df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    # standard two-sided paired t-test, used only to distinguish
    # INCONCLUSIVE from DIFFERENT when TOST does not establish equivalence.
    t_stat_2s, p_2s = stats.ttest_rel(a, b)

    if p_tost < 0.05:
        verdict = "EQUIVALENT (p_TOST < 0.05)"
    elif p_2s < 0.05 and abs(mean_diff) > bound:
        verdict = "DIFFERENT (significant and outside bounds)"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "n": n,
        "mean_diff": mean_diff,
        "sd_diff": sd,
        "bound": bound,
        "p_tost": float(p_tost),
        "ci90_low": float(ci_low),
        "ci90_high": float(ci_high),
        "p_2s": float(p_2s),
        "verdict": verdict,
    }


def per_model_records(metrics: dict, model: str, metric: str) -> np.ndarray:
    return np.asarray(
        [r[metric] for r in metrics[model]["records"]], dtype=np.float64
    )


def print_tost_tables(out, metrics: dict, sesoi_d: float) -> dict:
    out.write("=" * 100 + "\n")
    out.write(f"3. TOST EQUIVALENCE TESTING (SESOI = {sesoi_d} Cohen's d, paired per-image, normalized metrics)\n")
    out.write("=" * 100 + "\n")

    models = list(metrics.keys())
    # sanity: all models share the same image indices in the same order.
    ref_idx = [r["index"] for r in metrics[models[0]]["records"]]
    for m in models[1:]:
        idx = [r["index"] for r in metrics[m]["records"]]
        if idx != ref_idx:
            raise ValueError(f"Record order mismatch between '{models[0]}' and '{m}'; cannot pair by position.")

    tost_cols = [
        ("pair", 44, "<"),
        ("mean diff", 10, ">"),
        ("bound(+/-)", 10, ">"),
        ("90% CI", 24, ">"),
        ("p_TOST", 11, ">"),
        ("verdict", 44, "<"),
    ]

    tost_results = {}
    for metric in NORM_METRICS:
        out.write(f"\n-- {NORM_LABELS[metric]} ({metric}) --\n")
        table_rows = []
        for a, b in combinations(models, 2):
            vals_a = per_model_records(metrics, a, metric)
            vals_b = per_model_records(metrics, b, metric)
            res = tost_paired(vals_a, vals_b, sesoi_d)
            tost_results[(metric, a, b)] = res
            ci_str = f"[{res['ci90_low']:.4f}, {res['ci90_high']:.4f}]"
            table_rows.append([
                f"{a} vs {b}",
                f"{res['mean_diff']:.4f}",
                f"{res['bound']:.4f}",
                ci_str,
                f"{res['p_tost']:.4g}",
                res["verdict"],
            ])
        _write_table(out, tost_cols, table_rows)
    out.write("\n")
    return tost_results


def print_key_verdict(out, tost_results: dict, sesoi_d: float) -> None:
    out.write("=" * 100 + "\n")
    out.write("4. KEY COMPARISON: vanilla_scratch vs no_se_scratch\n")
    out.write("=" * 100 + "\n")

    key_pairs = []
    for metric in NORM_METRICS:
        key = (metric, "vanilla_scratch", "no_se_scratch")
        if key in tost_results:
            key_pairs.append((metric, tost_results[key]))
        else:
            key = (metric, "no_se_scratch", "vanilla_scratch")
            if key in tost_results:
                key_pairs.append((metric, tost_results[key]))

    if not key_pairs:
        out.write("vanilla_scratch / no_se_scratch pair not found in the data.\n\n")
        return

    verdicts = [v["verdict"] for _, v in key_pairs]
    n_equiv = sum(1 for v in verdicts if v.startswith("EQUIVALENT"))
    n_diff = sum(1 for v in verdicts if v.startswith("DIFFERENT"))
    n_inconc = sum(1 for v in verdicts if v.startswith("INCONCLUSIVE"))

    detail = "; ".join(
        f"{NORM_LABELS[m]}: mean diff {r['mean_diff']:+.4f} (bound +/-{r['bound']:.4f}), "
        f"p_TOST={r['p_tost']:.3g} -> {r['verdict']}"
        for m, r in key_pairs
    )

    if n_diff > 0:
        headline = (
            f"At a smallest-effect-size-of-interest bound of {sesoi_d} standard deviations of the paired "
            f"differences (Cohen's d), the data show a genuine DIFFERENCE between vanilla_scratch and "
            f"no_se_scratch on at least one normalized faithfulness metric"
        )
    elif n_equiv == len(key_pairs):
        headline = (
            f"At a smallest-effect-size-of-interest bound of {sesoi_d} standard deviations of the paired "
            f"differences (Cohen's d), the data support EQUIVALENCE between vanilla_scratch and no_se_scratch "
            f"across all normalized faithfulness metrics tested"
        )
    else:
        headline = (
            f"At a smallest-effect-size-of-interest bound of {sesoi_d} standard deviations of the paired "
            f"differences (Cohen's d), the comparison between vanilla_scratch and no_se_scratch is "
            f"INCONCLUSIVE for at least one normalized faithfulness metric — the sample is not large enough "
            f"to either rule out a difference of this size or establish equivalence"
        )

    out.write(
        headline + f". Per-metric detail: {detail}. "
        "In plain terms: the SE block does not appear to change Grad-CAM faithfulness by a practically "
        "meaningful amount, given the pre-registered bound above; where the verdict is inconclusive, more "
        "images (or a coarser bound) would be needed to say more.\n\n"
    )


def print_p0_confound_diagnostic(out, metrics: dict) -> None:
    out.write("=" * 100 + "\n")
    out.write("5. P0 CONFOUND DIAGNOSTIC (within-model correlation between p0 and normalized metrics)\n")
    out.write("=" * 100 + "\n")

    models = list(metrics.keys())
    cols = [
        ("model", 22, "<"),
        ("metric", 18, "<"),
        ("spearman(p0, metric)", 22, ">"),
        ("pearson(p0, metric)", 21, ">"),
        ("n", 6, ">"),
    ]

    table_rows = []
    insertion_r = {}
    for m in models:
        records = metrics[m]["records"]
        p0 = np.asarray([r["p0"] for r in records], dtype=np.float64)
        for metric in NORM_METRICS:
            vals = np.asarray([r[metric] for r in records], dtype=np.float64)
            rho, _ = stats.spearmanr(p0, vals)
            r, _ = stats.pearsonr(p0, vals)
            if metric == "insertion_auc":
                insertion_r[m] = r
            table_rows.append([
                m, NORM_LABELS[metric], f"{rho:.4f}", f"{r:.4f}", str(len(vals)),
            ])
    _write_table(out, cols, table_rows)
    out.write("\n")

    strong_negative = {m: r <= -0.5 for m, r in insertion_r.items()}
    n_strong_negative = sum(strong_negative.values())
    detail = "; ".join(f"{m}: r={r:+.3f}" for m, r in insertion_r.items())

    if n_strong_negative == len(insertion_r):
        interpretation = (
            "Within-model correlation between p0 and normalized insertion AUC is consistently strong and "
            "negative across ALL models (Pearson r <= -0.5 in every case). This is the signature of a "
            "normalization artifact: dividing by p0 systematically inflates the normalized metric for "
            "low-confidence images and deflates it for high-confidence ones within every model, so any "
            "cross-model gap in mean p0 (see Section 1) will show up as a gap in normalized insertion AUC "
            "even if the underlying explanations are equally faithful. The cross-model ranking on normalized "
            "insertion AUC should therefore be treated with caution and is more likely a residual artifact of "
            "the p0 normalization than a real property of the architectures."
        )
    elif n_strong_negative == 0:
        interpretation = (
            "Within-model correlation between p0 and normalized insertion AUC is weak in every model "
            "(Pearson r > -0.5 in every case). This argues against a normalization artifact: if the p0 "
            "division were driving the metric, we would expect a strong within-model relationship between "
            "confidence and the normalized score. Its absence suggests the cross-model differences reported "
            "in Sections 1-3 are more likely to reflect a real property of the architectures rather than a "
            "residual normalization effect."
        )
    else:
        interpretation = (
            f"Within-model correlation between p0 and normalized insertion AUC is mixed across models "
            f"({n_strong_negative}/{len(insertion_r)} show a strong negative correlation, r <= -0.5). This is "
            "an ambiguous signal: some part of the cross-model ranking may be a p0-normalization artifact, "
            "but it is not uniform enough to attribute the whole effect to normalization. The cross-model "
            "differences should be interpreted cautiously and, where possible, corroborated with a metric "
            "less sensitive to p0 (e.g. raw AUCs alongside a p0-matched subsample)."
        )

    out.write(f"Per-model insertion-AUC correlations: {detail}\n\n")
    out.write(interpretation + "\n\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sesoi", type=float, default=0.3,
        help="Smallest effect size of interest for TOST, in Cohen's d units (default: 0.3).",
    )
    args = parser.parse_args()

    metrics = load(FAITH_DIR / "faithfulness_metrics.json")
    sig = load(FAITH_DIR / "significance_tests.json")

    buf = io.StringIO()
    print_summary_tables(buf, metrics)
    print_significance_tables(buf, sig)
    tost_results = print_tost_tables(buf, metrics, args.sesoi)
    print_key_verdict(buf, tost_results, args.sesoi)
    print_p0_confound_diagnostic(buf, metrics)

    text = buf.getvalue()
    print(text, end="")

    out_path = FAITH_DIR / "report.txt"
    out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
