"""Phase 7.1: CAM concentration confound diagnostic + formal drift equivalence.

PART A -- is the Phase 7 explanation-drift ranking across models actually just a
byproduct of CAM sharpness (a more peaked CAM has "more room" to move, purely as
a geometric artifact of concentration, independent of anything architectural)?
For the SAME 200 clean images used in Phase 7, we compute three per-image CAM
concentration measures (normalized entropy, Gini coefficient, top-20%-mass
fraction), then correlate each, WITHIN every model separately, against that
model's per-image mean explanation drift (from runs/robustness/robustness_metrics.json,
not recomputed). Consistently strong within-model correlations across every model
would mean the cross-model drift ranking is substantially a sharpness artifact;
weak within-model correlations would mean the cross-model differences are real.

PART B -- makes the Phase 6.2-style drift null formal: TOST equivalence testing
(SESOI = 0.3 Cohen's d, 90% CI, three-way verdict) on the Phase 7 per-image mean
drift, for every model pair, on all four drift metrics -- so we can say whether
vanilla_scratch vs no_se_scratch is EQUIVALENT on explanation drift, not merely
"negligible d".

Recomputes Grad-CAMs only for the Part A concentration measures; Part B reads
runs/robustness/robustness_metrics.json directly.

Example (PowerShell):

    python scripts/concentration_diagnostic.py `
        --checkpoints runs/vanilla_scratch/checkpoints/best.pth `
        runs/no_se_scratch/checkpoints/best.pth `
        runs/small_kernel_scratch/checkpoints/best.pth `
        runs/vanilla_finetune/checkpoints/best.pth
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from src.data import build_loaders
from src.explain.gradcam import GradCAM

from report_faithfulness import tost_paired
from robustness_eval import (
    DRIFT_METRICS,
    SyntheticTestSet,
    drift_score,
    load_model,
    resolve_device,
)

CONCENTRATION_MEASURES = ("norm_entropy", "gini", "top20_mass_frac")
# For all three, direction relative to "more concentrated" differs: entropy is
# LOWER when more concentrated; gini and top20_mass_frac are HIGHER.
CONCENTRATION_LABELS = {
    "norm_entropy": "norm entropy (lower = more concentrated)",
    "gini": "Gini coefficient (higher = more concentrated)",
    "top20_mass_frac": "top-20% mass fraction (higher = more concentrated)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Same checkpoints (same order) as Phase 7.")
    parser.add_argument("--robustness-metrics", default="runs/robustness/robustness_metrics.json")
    parser.add_argument("--output-dir", default="runs/robustness")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true", help="Use a synthetic random test set instead of CIFAR-10.")
    parser.add_argument("--sesoi", type=float, default=0.3, help="Smallest effect size of interest for TOST, Cohen's d.")
    return parser.parse_args()


def normalized_entropy(cam: np.ndarray, eps: float = 1e-12) -> float:
    p = cam.ravel().astype(np.float64)
    total = p.sum()
    if total <= eps:
        return 1.0
    p = p / total
    p = p[p > eps]
    ent = -np.sum(p * np.log(p))
    max_ent = np.log(cam.size)
    return float(ent / max_ent) if max_ent > 0 else 0.0


def gini_coefficient(cam: np.ndarray) -> float:
    x = np.sort(cam.ravel().astype(np.float64))
    n = x.size
    total = x.sum()
    if total <= 1e-12:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def top20_mass_fraction(cam: np.ndarray) -> float:
    flat = cam.ravel().astype(np.float64)
    total = flat.sum()
    if total <= 1e-12:
        return 0.0
    k = max(1, int(round(0.20 * flat.size)))
    top_sum = np.sort(flat)[-k:].sum()
    return float(top_sum / total)


def compute_concentration(cam: np.ndarray) -> dict:
    return {
        "norm_entropy": normalized_entropy(cam),
        "gini": gini_coefficient(cam),
        "top20_mass_frac": top20_mass_fraction(cam),
    }


def load_shared_indices(robustness_metrics: dict) -> list:
    names = list(robustness_metrics.keys())
    ref_indices = sorted({r["index"] for r in robustness_metrics[names[0]]["records"]})
    for name in names[1:]:
        idx = sorted({r["index"] for r in robustness_metrics[name]["records"]})
        if idx != ref_indices:
            raise ValueError(f"Image index mismatch between '{names[0]}' and '{name}'; Phase 7 was not paired.")
    return ref_indices


def compute_cams(model, dataset, indices: list, device: torch.device, desc: str) -> dict:
    """Clean-image Grad-CAM per index (no corruption -- Part A only needs the
    clean CAMs already implicit in Phase 7's paired design)."""
    cams = {}
    for idx in tqdm(indices, desc=desc):
        image, _ = dataset[idx]
        image = image.to(device)
        with GradCAM(model) as gradcam:
            cam, _ = gradcam(image.unsqueeze(0))
        cams[idx] = cam[0].numpy()
    return cams


def per_image_mean_drift(records_by_model: dict, metric: str) -> dict:
    """name -> (sorted_indices, np.array of per-image mean drift over all
    corruption/severity combos), same construction as robustness_eval's
    compute_significance."""
    result = {}
    for name, records in records_by_model.items():
        by_idx = defaultdict(list)
        for r in records:
            by_idx[r["index"]].append(drift_score(r, metric))
        order = sorted(by_idx.keys())
        result[name] = (order, np.array([np.mean(by_idx[i]) for i in order]))
    return result


def _fmt_pm(vals: np.ndarray) -> str:
    return f"{np.mean(vals):.4f}+/-{np.std(vals):.4f}"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    robustness_path = Path(args.robustness_metrics)
    with open(robustness_path) as f:
        robustness_metrics = json.load(f)
    records_by_model = {name: robustness_metrics[name]["records"] for name in robustness_metrics}

    shared_indices = load_shared_indices(robustness_metrics)
    print(f"Loaded {len(shared_indices)} shared image indices from {robustness_path}.")

    if args.no_download:
        base_dataset = SyntheticTestSet(n=max(512, len(shared_indices)), seed=args.seed)
    else:
        _, test_loader = build_loaders(root=args.data_root, num_workers=0, download=True)
        base_dataset = test_loader.dataset

    if len(args.checkpoints) != len(robustness_metrics):
        raise ValueError(
            f"Got {len(args.checkpoints)} checkpoints but {robustness_path} has "
            f"{len(robustness_metrics)} models; pass the same set/order as Phase 7."
        )

    concentration_by_model = {}
    concentration_records_by_model = {}
    for ckpt in args.checkpoints:
        model, name = load_model(Path(ckpt), device)
        if name not in records_by_model:
            raise ValueError(f"Checkpoint '{ckpt}' resolves to model name '{name}' not found in {robustness_path}.")
        model.to(device)

        cams = compute_cams(model, base_dataset, shared_indices, device, desc=f"concentration[{name}]")
        per_measure = {measure: [] for measure in CONCENTRATION_MEASURES}
        per_image_records = []
        for idx in shared_indices:
            conc = compute_concentration(cams[idx])
            per_image_records.append({"index": idx, **conc})
            for measure in CONCENTRATION_MEASURES:
                per_measure[measure].append(conc[measure])
        concentration_by_model[name] = {measure: np.array(vals) for measure, vals in per_measure.items()}
        concentration_records_by_model[name] = per_image_records

    buf = io.StringIO()

    # --- Part A.1: summary table ---
    buf.write("=" * 100 + "\n")
    buf.write("PART A.1: PER-MODEL CAM CONCENTRATION SUMMARY (mean +/- std over shared images)\n")
    buf.write("=" * 100 + "\n")
    header = f"{'model':<20}{'norm entropy':>20}{'Gini':>20}{'top20% mass':>20}"
    buf.write(header + "\n")
    buf.write("-" * len(header) + "\n")
    for name, per_measure in concentration_by_model.items():
        row = (
            f"{name:<20}"
            f"{_fmt_pm(per_measure['norm_entropy']):>20}"
            f"{_fmt_pm(per_measure['gini']):>20}"
            f"{_fmt_pm(per_measure['top20_mass_frac']):>20}"
        )
        buf.write(row + "\n")
    buf.write(
        "\n(expect vanilla_finetune to show the LOWEST norm entropy and HIGHEST Gini / top20% mass -- "
        "i.e. the most concentrated CAMs, consistent with Phase 6's faithfulness findings.)\n\n"
    )

    # --- Part A.2/A.3: within-model correlation, concentration vs drift ---
    spearman_drift = per_image_mean_drift(records_by_model, "spearman")
    iou_drift = per_image_mean_drift(records_by_model, "top_k_iou")

    def _write_correlation_table(drift_by_model: dict, drift_label: str) -> dict:
        buf.write("=" * 100 + "\n")
        buf.write(f"WITHIN-MODEL CORRELATION: CAM concentration vs per-image mean drift ({drift_label})\n")
        buf.write("=" * 100 + "\n")
        cols_header = f"{'model':<20}{'measure':<32}{'spearman r':>14}{'pearson r':>14}{'n':>6}"
        buf.write(cols_header + "\n")
        buf.write("-" * len(cols_header) + "\n")
        r_by_model_measure = {}
        for name, per_measure in concentration_by_model.items():
            drift_idx, drift_vals = drift_by_model[name]
            if drift_idx != shared_indices:
                raise ValueError(f"Index mismatch between concentration and drift data for '{name}'.")
            for measure in CONCENTRATION_MEASURES:
                conc_vals = per_measure[measure]
                rho, _ = stats.spearmanr(conc_vals, drift_vals)
                r, _ = stats.pearsonr(conc_vals, drift_vals)
                r_by_model_measure[(name, measure)] = (rho, r)
                buf.write(
                    f"{name:<20}{CONCENTRATION_LABELS[measure]:<32}{rho:>14.4f}{r:>14.4f}{len(drift_vals):>6}\n"
                )
        buf.write("\n")
        return r_by_model_measure

    r_spearman_drift = _write_correlation_table(spearman_drift, "1 - spearman similarity")
    r_iou_drift = _write_correlation_table(iou_drift, "1 - top-k IoU")

    def _strength_summary(r_dict: dict, measure: str) -> list:
        return [r_dict[(name, measure)][1] for name in concentration_by_model]

    primary_measure = "top20_mass_frac"
    primary_r_spearman = _strength_summary(r_spearman_drift, primary_measure)
    primary_r_iou = _strength_summary(r_iou_drift, primary_measure)

    buf.write("=" * 100 + "\n")
    buf.write("PART A.2: KEY TEST -- is the cross-model drift ranking a sharpness artifact?\n")
    buf.write("=" * 100 + "\n")
    detail_spearman = "; ".join(
        f"{name}: r={r:+.3f}" for name, r in zip(concentration_by_model.keys(), primary_r_spearman)
    )
    buf.write(
        f"Per-model Pearson r between {CONCENTRATION_LABELS[primary_measure]} and (1 - spearman) drift: "
        f"{detail_spearman}\n\n"
    )
    all_strong_spearman = all(abs(r) >= 0.5 for r in primary_r_spearman)
    all_weak_spearman = all(abs(r) < 0.5 for r in primary_r_spearman)
    if all_strong_spearman:
        interp_a = (
            "WITHIN every model, more concentrated CAMs drift more (|r| >= 0.5 in every case, on spearman-based "
            "drift). This is the signature of a sharpness artifact: the cross-model drift ranking reported in "
            "Phase 7 is substantially explained by which model happens to produce peakier CAMs, not by an "
            "architectural difference in explanation stability."
        )
    elif all_weak_spearman:
        interp_a = (
            "WITHIN every model, concentration and spearman-based drift are only weakly correlated (|r| < 0.5 in "
            "every case). This argues the cross-model drift differences reported in Phase 7 are real -- not an "
            "artifact of CAM sharpness."
        )
    else:
        interp_a = (
            "Within-model correlation between concentration and spearman-based drift is MIXED across models -- "
            "some show a strong relationship, others do not. The cross-model spearman-drift ranking should be "
            "treated cautiously: part of it may be a sharpness artifact, but not uniformly enough to attribute "
            "the whole effect to it."
        )
    buf.write(interp_a + "\n\n")

    buf.write("=" * 100 + "\n")
    buf.write("PART A.3: REPEAT USING TOP-20% IoU DRIFT\n")
    buf.write("=" * 100 + "\n")
    detail_iou = "; ".join(f"{name}: r={r:+.3f}" for name, r in zip(concentration_by_model.keys(), primary_r_iou))
    buf.write(
        f"Per-model Pearson r between {CONCENTRATION_LABELS[primary_measure]} and (1 - top-k IoU) drift: "
        f"{detail_iou}\n\n"
    )
    mean_abs_r_spearman = float(np.mean(np.abs(primary_r_spearman)))
    mean_abs_r_iou = float(np.mean(np.abs(primary_r_iou)))
    buf.write(
        f"Mean |r| across models: spearman-drift = {mean_abs_r_spearman:.3f}, IoU-drift = {mean_abs_r_iou:.3f}.\n\n"
    )
    iou_much_less_correlated = mean_abs_r_iou < mean_abs_r_spearman - 0.15
    if iou_much_less_correlated:
        interp_b = (
            "Top-20% IoU drift is markedly LESS correlated with CAM concentration than spearman-based drift is. "
            "This argues top-k IoU is the more trustworthy cross-model drift metric: it is less confounded by "
            "how peaked a model's CAMs happen to be, so differences in IoU drift are more likely to reflect a "
            "real property of the explanations rather than a side effect of sharpness."
        )
    else:
        interp_b = (
            "Top-20% IoU drift is not markedly less correlated with CAM concentration than spearman-based drift "
            "is; the sharpness-confound concern applies comparably to both metrics."
        )
    buf.write(interp_b + "\n\n")

    buf.write("=" * 100 + "\n")
    buf.write("PART A.4: RECOMMENDATION\n")
    buf.write("=" * 100 + "\n")
    if all_strong_spearman and iou_much_less_correlated:
        recommendation = (
            "Use TOP-20% IoU DRIFT as the paper's headline explanation-stability metric. Spearman-based drift is "
            "substantially confounded by CAM concentration (a sharpness artifact common to every model), while "
            "IoU drift is comparatively immune to it."
        )
    elif all_weak_spearman:
        recommendation = (
            "Spearman-based drift is not meaningfully confounded by CAM concentration in this data, so it can "
            "remain the paper's headline explanation-stability metric; top-k IoU drift is a reasonable "
            "corroborating secondary metric."
        )
    else:
        recommendation = (
            "Given the mixed/partial confound evidence, report BOTH spearman-based and top-k IoU drift as "
            "headline metrics side by side, and flag the concentration confound explicitly rather than picking "
            "a single number."
        )
    buf.write(recommendation + "\n\n")

    # --- Part B: TOST equivalence testing on Phase 7 drift ---
    buf.write("=" * 100 + "\n")
    buf.write(f"PART B: TOST EQUIVALENCE TESTING ON EXPLANATION DRIFT (SESOI = {args.sesoi} Cohen's d, paired per-image)\n")
    buf.write("=" * 100 + "\n")

    tost_cols_header = f"{'pair':<44}{'mean diff':>10}{'bound(+/-)':>11}{'90% CI':>24}{'p_TOST':>11}  verdict"
    tost_results = {}
    for metric in DRIFT_METRICS:
        drift_by_model = per_image_mean_drift(records_by_model, metric)
        buf.write(f"\n-- {metric} drift --\n")
        buf.write(tost_cols_header + "\n")
        buf.write("-" * len(tost_cols_header) + "\n")
        names = list(drift_by_model.keys())
        for a, b in combinations(names, 2):
            idx_a, vals_a = drift_by_model[a]
            idx_b, vals_b = drift_by_model[b]
            if idx_a != idx_b:
                raise ValueError(f"Index mismatch between '{a}' and '{b}' for metric '{metric}'.")
            res = tost_paired(vals_a, vals_b, args.sesoi)
            tost_results[(metric, a, b)] = res
            ci_str = f"[{res['ci90_low']:.4f}, {res['ci90_high']:.4f}]"
            buf.write(
                f"{a + ' vs ' + b:<44}{res['mean_diff']:>10.4f}{res['bound']:>11.4f}{ci_str:>24}"
                f"{res['p_tost']:>11.4g}  {res['verdict']}\n"
            )
    buf.write("\n")

    buf.write("=" * 100 + "\n")
    buf.write("PART B KEY VERDICT: vanilla_scratch vs no_se_scratch on explanation drift\n")
    buf.write("=" * 100 + "\n")
    key_pairs = []
    for metric in DRIFT_METRICS:
        key = (metric, "vanilla_scratch", "no_se_scratch")
        if key in tost_results:
            key_pairs.append((metric, tost_results[key]))
        else:
            key_alt = (metric, "no_se_scratch", "vanilla_scratch")
            if key_alt in tost_results:
                key_pairs.append((metric, tost_results[key_alt]))

    if not key_pairs:
        buf.write("vanilla_scratch / no_se_scratch pair not found in the data.\n\n")
    else:
        verdicts = [v["verdict"] for _, v in key_pairs]
        n_equiv = sum(1 for v in verdicts if v.startswith("EQUIVALENT"))
        n_diff = sum(1 for v in verdicts if v.startswith("DIFFERENT"))
        detail = "; ".join(
            f"{m} drift: mean diff {r['mean_diff']:+.4f} (bound +/-{r['bound']:.4f}), "
            f"p_TOST={r['p_tost']:.3g} -> {r['verdict']}"
            for m, r in key_pairs
        )
        if n_diff > 0:
            headline = (
                f"At SESOI={args.sesoi} Cohen's d, the data show a genuine DIFFERENCE between vanilla_scratch and "
                f"no_se_scratch on at least one explanation-drift metric"
            )
        elif n_equiv == len(key_pairs):
            headline = (
                f"At SESOI={args.sesoi} Cohen's d, the data support EQUIVALENCE between vanilla_scratch and "
                f"no_se_scratch across ALL FOUR explanation-drift metrics -- this is a formal statistical claim, "
                f"not merely a small observed Cohen's d"
            )
        else:
            headline = (
                f"At SESOI={args.sesoi} Cohen's d, the comparison between vanilla_scratch and no_se_scratch is "
                f"INCONCLUSIVE for at least one explanation-drift metric -- the sample is not large enough to "
                f"either rule out a difference of this size or establish equivalence"
            )
        buf.write(headline + f". Per-metric detail: {detail}.\n\n")

    text = buf.getvalue()
    print(text, end="")

    report_path = output_dir / "concentration_report.txt"
    report_path.write_text(text, encoding="utf-8")
    print(f"Saved report to {report_path.resolve()}")

    concentration_out = {
        name: {
            "aggregate": {measure: {"mean": float(np.mean(vals)), "std": float(np.std(vals))} for measure, vals in per_measure.items()},
            "records": concentration_records_by_model[name],
        }
        for name, per_measure in concentration_by_model.items()
    }
    concentration_path = output_dir / "concentration_metrics.json"
    with open(concentration_path, "w") as f:
        json.dump(concentration_out, f, indent=2)
    print(f"Saved per-model concentration metrics to {concentration_path.resolve()}")

    tost_out = {
        f"{metric}|{a}|{b}": res for (metric, a, b), res in tost_results.items()
    }
    tost_path = output_dir / "drift_equivalence_tests.json"
    with open(tost_path, "w") as f:
        json.dump(tost_out, f, indent=2)
    print(f"Saved drift TOST equivalence tests to {tost_path.resolve()}")


if __name__ == "__main__":
    main()
