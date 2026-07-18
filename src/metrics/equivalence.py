"""Two one-sided tests (TOST) for statistical equivalence.

Used by scripts/report_faithfulness.py, scripts/report_robustness.py, and
scripts/concentration_diagnostic.py to test whether two models' paired
per-image metrics are equivalent (rather than merely "not significantly
different") within a smallest-effect-size-of-interest (SESOI) bound.
"""

import math
from typing import Dict

import numpy as np
from scipy import stats

# Default smallest effect size of interest, in Cohen's d, used across the
# faithfulness (Phase 6.2), robustness (Phase 7.1), and concentration
# diagnostics unless a script overrides it via --sesoi.
DEFAULT_SESOI_D = 0.3


def tost_paired(a: np.ndarray, b: np.ndarray, sesoi_d: float = DEFAULT_SESOI_D) -> Dict:
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
