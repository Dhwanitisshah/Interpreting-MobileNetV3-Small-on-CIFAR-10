"""Phase 7.2: per-corruption breakdown of Phase 7's explanation-drift results.

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

Output goes to stdout and to runs/robustness/report.txt.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROBUSTNESS_DIR = ROOT / "runs" / "robustness"

BASELINE_MODELS = ("vanilla_scratch", "no_se_scratch")
HIGH_SHIFT_CORRUPTIONS = ("brightness", "contrast")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    metrics = load(ROBUSTNESS_DIR / "robustness_metrics.json")
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

    text = buf.getvalue()
    print(text, end="")

    out_path = ROBUSTNESS_DIR / "report.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved report to {out_path.resolve()}")

    breakdown_path = ROBUSTNESS_DIR / "per_corruption_breakdown.json"
    with open(breakdown_path, "w") as f:
        json.dump({"corruptions": corruptions, "severities": severities, "uniformity": uniformity}, f, indent=2)
    print(f"Saved per-corruption breakdown data to {breakdown_path.resolve()}")


if __name__ == "__main__":
    main()
