"""Phase 7.2/7.4: per-corruption breakdown of Phase 7's explanation-drift results,
plus an accuracy-floor sensitivity analysis.

Reads runs/robustness/robustness_metrics.json only (no recomputation, no
recomputed CAMs) and prints:
  1. Model x corruption tables of mean spearman drift, one per severity.
  2. Model x corruption tables of accuracy under corruption, one per severity.
  3. A diagnostic: is vanilla_finetune's elevated drift (see Phase 7's summary
     table) spread uniformly across all six corruptions, or concentrated in
     specific families -- especially brightness/contrast, which shift input
     statistics furthest from the ImageNet normalization the model was trained
     under?
  4. The same per-corruption view for the accuracy drop, so drift and accuracy
     degradation can be compared corruption-by-corruption.
  5. A plain-English interpretation of whether the drift/accuracy dissociation
     found in Phase 7 is a general property of the model or driven by
     particular corruption families.
  6. (Phase 7.4, --min-accuracy) A sensitivity analysis that drops every
     (corruption, severity) cell where ANY model's accuracy_under_corruption
     falls below the threshold, then recomputes per-model mean spearman drift,
     the per-corruption excess-drift table, and the key TOST verdicts, printed
     side-by-side against the unfiltered numbers. Rationale: under
     gaussian_noise at severities 3 and 5, every model sits at exactly 0.100
     accuracy -- chance for 10 classes. A model that is guessing uniformly has
     no interpretable "explanation" for its prediction, so explanation drift
     measured there is not meaningful on its own terms, and the paper should
     report whether its drift conclusions survive dropping those cells.

Output goes to stdout and to runs/robustness/report.txt (or <robustness-dir>/report.txt).
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_faithfulness import tost_paired  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROBUSTNESS_DIR = ROOT / "runs" / "robustness"

BASELINE_MODELS = ("vanilla_scratch", "no_se_scratch")
HIGH_SHIFT_CORRUPTIONS = ("brightness", "contrast")
SESOI_D = 0.3  # Cohen's d, same convention as Phase 6.2/7.1


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_agg_keys(aggregate: dict):
    corruptions, severities = [], []
    for key in aggregate:
        corruption, sev_part = key.rsplit("|sev", 1)
        if corruption not in corruptions:
            corruptions.append(corruption)
        sev = int(sev_part)
        if sev not in severities:
            severities.append(sev)
    return corruptions, sorted(severities)


def _write_table(out, cols, rows) -> None:
    header = "  ".join(f"{name:{align}{width}}" for name, width, align in cols)
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")
    for cells in rows:
        out.write("  ".join(f"{cell:{align}{width}}" for cell, (_, width, align) in zip(cells, cols)) + "\n")


def print_breakdown_tables(out, metrics: dict, corruptions: list, severities: list, field: str, label: str, xform=None) -> None:
    """One table per severity: model x corruption, values from `field` in the
    per-model aggregate (optionally transformed, e.g. 1 - mean_spearman)."""
    models = list(metrics.keys())
    xform = xform or (lambda v: v)

    out.write("=" * 100 + "\n")
    out.write(f"{label}\n")
    out.write("=" * 100 + "\n")
    for sev in severities:
        out.write(f"\n-- severity {sev} --\n")
        cols = [("model", 22, "<")] + [(c, 14, ">") for c in corruptions]
        rows = []
        for m in models:
            agg = metrics[m]["aggregate"]
            row = [m]
            for c in corruptions:
                key = f"{c}|sev{sev}"
                row.append(f"{xform(agg[key][field]):.4f}" if key in agg else "n/a")
            rows.append(row)
        _write_table(out, cols, rows)
    out.write("\n")


def per_corruption_mean(metrics: dict, model: str, corruption: str, severities: list, field: str, xform) -> float:
    agg = metrics[model]["aggregate"]
    vals = [xform(agg[f"{corruption}|sev{sev}"][field]) for sev in severities if f"{corruption}|sev{sev}" in agg]
    return float(np.mean(vals)) if vals else float("nan")


def print_uniformity_diagnostic(out, metrics: dict, corruptions: list, severities: list) -> dict:
    out.write("=" * 100 + "\n")
    out.write("UNIFORMITY DIAGNOSTIC: is vanilla_finetune's elevated drift spread across all corruptions,\n")
    out.write("or concentrated in specific families (esp. brightness/contrast)?\n")
    out.write("=" * 100 + "\n")

    models = list(metrics.keys())
    if "vanilla_finetune" not in models:
        out.write("vanilla_finetune not found in the data; skipping.\n\n")
        return {}

    baselines_present = [m for m in BASELINE_MODELS if m in models]
    if not baselines_present:
        out.write("No baseline scratch models found; skipping.\n\n")
        return {}

    drift_xform = lambda v: 1.0 - v  # noqa: E731

    per_corruption_excess = {}
    for c in corruptions:
        ft_drift = per_corruption_mean(metrics, "vanilla_finetune", c, severities, "mean_spearman", drift_xform)
        baseline_drift = float(
            np.mean([per_corruption_mean(metrics, b, c, severities, "mean_spearman", drift_xform) for b in baselines_present])
        )
        per_corruption_excess[c] = ft_drift - baseline_drift

    cols = [("corruption", 20, "<"), ("vanilla_finetune drift", 24, ">"), ("baseline mean drift", 22, ">"), ("excess", 10, ">")]
    rows = []
    for c in corruptions:
        ft_drift = per_corruption_mean(metrics, "vanilla_finetune", c, severities, "mean_spearman", drift_xform)
        baseline_drift = float(
            np.mean([per_corruption_mean(metrics, b, c, severities, "mean_spearman", drift_xform) for b in baselines_present])
        )
        rows.append([c, f"{ft_drift:.4f}", f"{baseline_drift:.4f}", f"{per_corruption_excess[c]:+.4f}"])
    _write_table(out, cols, rows)
    out.write("\n")

    excess_vals = np.array(list(per_corruption_excess.values()))
    mean_excess = float(excess_vals.mean())
    std_excess = float(excess_vals.std())
    cv = std_excess / mean_excess if mean_excess > 1e-9 else float("inf")

    high_shift_present = [c for c in HIGH_SHIFT_CORRUPTIONS if c in per_corruption_excess]
    other_corruptions = [c for c in corruptions if c not in HIGH_SHIFT_CORRUPTIONS]
    high_shift_excess = float(np.mean([per_corruption_excess[c] for c in high_shift_present])) if high_shift_present else float("nan")
    other_excess = float(np.mean([per_corruption_excess[c] for c in other_corruptions])) if other_corruptions else float("nan")

    out.write(
        f"Mean excess drift (vanilla_finetune - baseline mean) across corruptions: {mean_excess:+.4f} "
        f"(std={std_excess:.4f}, coefficient of variation={cv:.2f}).\n"
    )
    out.write(
        f"Mean excess on brightness/contrast: {high_shift_excess:+.4f}; mean excess on the other four "
        f"corruptions: {other_excess:+.4f}.\n\n"
    )

    is_uniform = cv < 0.5
    high_shift_dominant = high_shift_present and other_corruptions and high_shift_excess > 1.5 * max(other_excess, 1e-9)

    if is_uniform and not high_shift_dominant:
        interp = (
            "The excess drift is roughly the SAME MAGNITUDE across all six corruptions (low coefficient of "
            "variation, and brightness/contrast are not disproportionately worse than the other four). This "
            "argues vanilla_finetune's elevated explanation drift is a GENERAL property of the model -- its "
            "explanations are broadly less stable under any distribution shift, not specifically sensitive to "
            "corruptions that push inputs furthest from the ImageNet normalization statistics."
        )
    elif high_shift_dominant:
        interp = (
            "The excess drift is CONCENTRATED in brightness/contrast, which is disproportionately larger than "
            "the excess on the other four corruptions. This argues vanilla_finetune's elevated explanation drift "
            "is driven by a specific corruption family -- corruptions that shift low-level input statistics "
            "(pixel intensity/contrast) furthest from the ImageNet normalization the pretrained backbone expects "
            "-- rather than being a uniformly general instability."
        )
    else:
        interp = (
            "The excess drift varies substantially across corruptions (high coefficient of variation) but is "
            "not clearly dominated by brightness/contrast specifically. This suggests the elevated drift is "
            "corruption-dependent rather than a single uniform property, but the specific pattern does not "
            "match the brightness/contrast hypothesis cleanly -- it should be treated as a mixed result."
        )
    out.write(interp + "\n\n")

    return {"excess_by_corruption": per_corruption_excess, "mean_excess": mean_excess, "std_excess": std_excess, "cv": cv}


def print_accuracy_diagnostic(out, metrics: dict, corruptions: list, severities: list) -> None:
    out.write("=" * 100 + "\n")
    out.write("ACCURACY VIEW: per-corruption accuracy under corruption (same models/corruptions as above)\n")
    out.write("=" * 100 + "\n")

    models = list(metrics.keys())
    cols = [("model", 22, "<")] + [(c, 14, ">") for c in corruptions]
    rows = []
    for m in models:
        row = [m]
        for c in corruptions:
            acc = per_corruption_mean(metrics, m, c, severities, "accuracy_under_corruption", lambda v: v)
            row.append(f"{acc:.4f}")
        rows.append(row)
    _write_table(out, cols, rows)
    out.write("\n(mean accuracy under corruption, averaged over severities, per corruption type.)\n\n")


def print_dissociation_interpretation(out, metrics: dict, corruptions: list, severities: list, uniformity: dict) -> None:
    out.write("=" * 100 + "\n")
    out.write("INTERPRETATION: is the drift/accuracy dissociation general or corruption-family-specific?\n")
    out.write("=" * 100 + "\n")

    if not uniformity:
        out.write("Insufficient data (vanilla_finetune or baseline models missing) to draw a conclusion.\n\n")
        return

    models = list(metrics.keys())
    if "vanilla_finetune" not in models:
        out.write("vanilla_finetune not found; skipping.\n\n")
        return

    baselines_present = [m for m in BASELINE_MODELS if m in models]
    per_corruption_acc_drop = {}
    for c in corruptions:
        ft_acc = per_corruption_mean(metrics, "vanilla_finetune", c, severities, "accuracy_under_corruption", lambda v: v)
        baseline_acc = float(
            np.mean([per_corruption_mean(metrics, b, c, severities, "accuracy_under_corruption", lambda v: v) for b in baselines_present])
        )
        per_corruption_acc_drop[c] = baseline_acc - ft_acc  # positive => finetune loses MORE accuracy than baselines

    excess_drift = uniformity["excess_by_corruption"]
    drift_rank = sorted(corruptions, key=lambda c: excess_drift[c], reverse=True)
    acc_rank = sorted(corruptions, key=lambda c: per_corruption_acc_drop[c], reverse=True)

    out.write(f"Corruptions ranked by vanilla_finetune's EXCESS drift (largest first): {drift_rank}\n")
    out.write(f"Corruptions ranked by vanilla_finetune's relative accuracy DISADVANTAGE (largest first): {acc_rank}\n\n")

    top_k = 3
    overlap = len(set(drift_rank[:top_k]) & set(acc_rank[:top_k]))

    if overlap >= 2:
        out.write(
            f"The corruptions with the largest excess drift substantially OVERLAP ({overlap}/{top_k} in the "
            "top ranks) with the corruptions where vanilla_finetune loses the most relative accuracy. This "
            "weakens the Phase 7 dissociation claim for those specific corruptions -- there, drift does track "
            "accuracy loss more closely -- but the diagnostic above (uniform vs. concentrated excess) is the "
            "more direct test of whether the OVERALL dissociation is general or family-specific.\n\n"
        )
    else:
        out.write(
            f"The corruptions with the largest excess drift do NOT substantially overlap ({overlap}/{top_k} in "
            "the top ranks) with the corruptions where vanilla_finetune loses the most relative accuracy. Drift "
            "and accuracy degradation rank corruptions differently, reinforcing Phase 7's finding that "
            "explanation instability is not simply a readout of prediction accuracy.\n\n"
        )

    if uniformity["cv"] < 0.5:
        out.write(
            "Combined with the uniformity diagnostic (low coefficient of variation in excess drift across "
            "corruptions), the dissociation between explanation drift and accuracy loss appears to be a GENERAL "
            "property of vanilla_finetune -- present across corruption families, not isolated to inputs whose "
            "low-level statistics shift furthest from ImageNet normalization.\n\n"
        )
    else:
        out.write(
            "Combined with the uniformity diagnostic (high coefficient of variation in excess drift across "
            "corruptions), the dissociation is better described as CORRUPTION-FAMILY-SPECIFIC rather than a "
            "uniform property of vanilla_finetune.\n\n"
        )


def find_dropped_cells(metrics: dict, corruptions: list, severities: list, min_accuracy: float) -> list:
    """A (corruption, severity) cell is dropped if ANY model's accuracy under
    it falls below `min_accuracy` -- dropped uniformly across all models so
    the paired per-image drift arrays used for TOST stay aligned. Returns a
    list of (corruption, severity, min_accuracy_in_cell), sorted worst-first."""
    dropped = []
    for c in corruptions:
        for s in severities:
            key = f"{c}|sev{s}"
            accs = [
                metrics[m]["aggregate"][key]["accuracy_under_corruption"]
                for m in metrics if key in metrics[m]["aggregate"]
            ]
            if accs and min(accs) < min_accuracy:
                dropped.append((c, s, float(min(accs))))
    dropped.sort(key=lambda row: row[2])
    return dropped


def per_image_mean_drift_filtered(metrics: dict, model: str, dropped_cells: set, metric: str = "spearman"):
    """(sorted_indices, per-image mean (1 - metric) drift), averaged only over
    (corruption, severity) cells NOT in `dropped_cells`."""
    by_idx = defaultdict(list)
    for r in metrics[model]["records"]:
        if (r["corruption"], r["severity"]) in dropped_cells:
            continue
        by_idx[r["index"]].append(1.0 - r[metric])
    order = sorted(by_idx.keys())
    return order, np.array([np.mean(by_idx[i]) for i in order])


def _retained_severities(dropped_cells: set, corruption: str, severities: list) -> list:
    return [s for s in severities if (corruption, s) not in dropped_cells]


def print_sensitivity_analysis(out, metrics: dict, corruptions: list, severities: list, min_accuracy: float) -> None:
    out.write("=" * 100 + "\n")
    out.write(f"6. ACCURACY-FLOOR SENSITIVITY ANALYSIS (--min-accuracy {min_accuracy})\n")
    out.write("=" * 100 + "\n")

    dropped = find_dropped_cells(metrics, corruptions, severities, min_accuracy)
    dropped_set = {(c, s) for c, s, _ in dropped}

    if min_accuracy <= 0.0:
        out.write("--min-accuracy is 0.0 (default): no cells excluded, sensitivity analysis is a no-op.\n\n")
        return

    if not dropped:
        out.write(f"No (corruption, severity) cell has any model's accuracy below {min_accuracy}; nothing to drop.\n\n")
        return

    out.write(f"Cells dropped (min accuracy across all models < {min_accuracy}):\n")
    for c, s, acc in dropped:
        out.write(f"  {c}|sev{s}: worst model accuracy = {acc:.4f} (chance for 10 classes = 0.100)\n")
    out.write("\n")

    models = list(metrics.keys())

    # --- per-model mean spearman drift: unfiltered vs filtered ---
    out.write("-- per-model mean spearman drift: unfiltered vs filtered --\n")
    cols = [("model", 22, "<"), ("unfiltered", 14, ">"), ("filtered", 14, ">"), ("delta", 10, ">")]
    rows = []
    unfiltered_drift, filtered_drift = {}, {}
    for m in models:
        _, vals_all = per_image_mean_drift_filtered(metrics, m, set())
        _, vals_filt = per_image_mean_drift_filtered(metrics, m, dropped_set)
        unfiltered_drift[m] = float(vals_all.mean())
        filtered_drift[m] = float(vals_filt.mean())
        rows.append([m, f"{unfiltered_drift[m]:.4f}", f"{filtered_drift[m]:.4f}", f"{filtered_drift[m] - unfiltered_drift[m]:+.4f}"])
    _write_table(out, cols, rows)
    out.write("\n")

    # --- per-corruption excess-drift table: unfiltered vs filtered ---
    if "vanilla_finetune" in models:
        baselines_present = [m for m in BASELINE_MODELS if m in models]
        if baselines_present:
            out.write("-- per-corruption excess drift (vanilla_finetune - baseline mean): unfiltered vs filtered --\n")
            cols = [("corruption", 20, "<"), ("unfiltered excess", 20, ">"), ("filtered excess", 18, ">"), ("severities kept", 20, "<")]
            rows = []
            for c in corruptions:
                kept = _retained_severities(dropped_set, c, severities)
                excess_unf = per_corruption_mean(metrics, "vanilla_finetune", c, severities, "mean_spearman", lambda v: 1.0 - v) - float(
                    np.mean([per_corruption_mean(metrics, b, c, severities, "mean_spearman", lambda v: 1.0 - v) for b in baselines_present])
                )
                if kept:
                    excess_filt = per_corruption_mean(metrics, "vanilla_finetune", c, kept, "mean_spearman", lambda v: 1.0 - v) - float(
                        np.mean([per_corruption_mean(metrics, b, c, kept, "mean_spearman", lambda v: 1.0 - v) for b in baselines_present])
                    )
                    filt_str = f"{excess_filt:+.4f}"
                else:
                    filt_str = "all dropped"
                rows.append([c, f"{excess_unf:+.4f}", filt_str, str(kept)])
            _write_table(out, cols, rows)
            out.write("\n")

    # --- TOST verdicts: unfiltered vs filtered, for the pairs of interest ---
    out.write("-- TOST verdicts on spearman drift (SESOI = {:.1f} Cohen's d): unfiltered vs filtered --\n".format(SESOI_D))
    cols = [("pair", 44, "<"), ("unfiltered verdict", 44, "<"), ("filtered verdict", 44, "<")]
    rows = []
    pairs_of_interest = []
    if "vanilla_scratch" in models and "no_se_scratch" in models:
        pairs_of_interest.append(("vanilla_scratch", "no_se_scratch"))
    if "vanilla_finetune" in models:
        for b in BASELINE_MODELS:
            if b in models:
                pairs_of_interest.append(("vanilla_finetune", b))

    for a, b in pairs_of_interest:
        idx_a_unf, vals_a_unf = per_image_mean_drift_filtered(metrics, a, set())
        idx_b_unf, vals_b_unf = per_image_mean_drift_filtered(metrics, b, set())
        res_unf = tost_paired(vals_a_unf, vals_b_unf, SESOI_D)

        idx_a_filt, vals_a_filt = per_image_mean_drift_filtered(metrics, a, dropped_set)
        idx_b_filt, vals_b_filt = per_image_mean_drift_filtered(metrics, b, dropped_set)
        res_filt = tost_paired(vals_a_filt, vals_b_filt, SESOI_D)

        rows.append([f"{a} vs {b}", res_unf["verdict"], res_filt["verdict"]])
    _write_table(out, cols, rows)
    out.write("\n")

    changed = [f"{a} vs {b}" for (a, b), row in zip(pairs_of_interest, rows) if row[1] != row[2]]
    if changed:
        out.write(
            f"VERDICT CHANGED for: {changed}. The conclusion for these pairs is sensitive to including the "
            "near-chance-accuracy cells -- treat it as less robust and report both numbers.\n\n"
        )
    else:
        out.write(
            "No TOST verdict changed after dropping the near-chance-accuracy cells -- the reported conclusions "
            "are not an artifact of including uninterpretable (guessing) predictions.\n\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robustness-dir", default=str(ROBUSTNESS_DIR),
        help="Directory holding robustness_metrics.json (default: runs/robustness). "
             "Point at runs/robustness_fixed to report on the Phase 7.3 fixed-target-class rerun.",
    )
    parser.add_argument(
        "--min-accuracy", type=float, default=0.0,
        help="Phase 7.4 sensitivity analysis: exclude (corruption, severity) cells where any model's "
             "accuracy_under_corruption falls below this threshold before recomputing drift/TOST (default: 0.0, no-op).",
    )
    args = parser.parse_args()
    robustness_dir = Path(args.robustness_dir)

    metrics = load(robustness_dir / "robustness_metrics.json")
    corruptions, severities = parse_agg_keys(metrics[list(metrics.keys())[0]]["aggregate"])

    buf = io.StringIO()
    print_breakdown_tables(
        buf, metrics, corruptions, severities, "mean_spearman", "1. MEAN SPEARMAN DRIFT, MODEL x CORRUPTION (per severity)",
        xform=lambda v: 1.0 - v,
    )
    print_breakdown_tables(
        buf, metrics, corruptions, severities, "accuracy_under_corruption",
        "2. ACCURACY UNDER CORRUPTION, MODEL x CORRUPTION (per severity)",
    )
    uniformity = print_uniformity_diagnostic(buf, metrics, corruptions, severities)
    print_accuracy_diagnostic(buf, metrics, corruptions, severities)
    print_dissociation_interpretation(buf, metrics, corruptions, severities, uniformity)
    print_sensitivity_analysis(buf, metrics, corruptions, severities, args.min_accuracy)

    text = buf.getvalue()
    print(text, end="")

    out_path = robustness_dir / "report.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved report to {out_path.resolve()}")

    breakdown_path = robustness_dir / "per_corruption_breakdown.json"
    with open(breakdown_path, "w") as f:
        json.dump({"corruptions": corruptions, "severities": severities, "uniformity": uniformity}, f, indent=2)
    print(f"Saved per-corruption breakdown data to {breakdown_path.resolve()}")


if __name__ == "__main__":
    main()
