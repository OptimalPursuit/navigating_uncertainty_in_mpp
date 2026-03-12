# paired_tests.py
#
# Usage:
#   1) Put your per-instance objective values in a CSV.
#   2) One row = one instance.
#   3) Columns = methods, e.g.:
#        instance,SAC_UVP_R,Pen_SAC_R,Lag_SAC_R,SP_NA
#        1,1510.2,1321.7,1004.1,1042.5
#        2,1488.9,1298.3, 995.0,1031.2
#        ...
#
#   4) Run:
#        python paired_tests.py results_test.csv
#
# Optional:
#   - You can also run it on a generalization CSV separately.
#   - Edit COMPARISONS below to match your column names.

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ----------------------------
# Configure your comparisons
# ----------------------------
COMPARISONS: List[Tuple[str, str]] = [
    ("SAC_UVP_R", "Pen_SAC_R"),
    ("SAC_UVP_R", "Lag_SAC_R"),
    ("SAC_UVP_R", "SP_NA"),
    ("PPO_UVP_R", "Pen_PPO_R"),
    ("PPO_UVP_R", "Lag_PPO_R"),
    ("PPO_UVP_R", "SP_NA"),
]

N_BOOT = 20000
CI_LEVEL = 0.95
RANDOM_SEED = 42


@dataclass
class TestResult:
    method_a: str
    method_b: str
    n: int
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    mean_diff: float          # mean(A - B)
    median_diff: float        # median(A - B)
    ci_low: float
    ci_high: float
    p_value: float
    statistic: float


def paired_bootstrap_ci(
    diffs: np.ndarray,
    level: float = 0.95,
    n_boot: int = 20000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap CI for the paired mean difference.
    """
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    alpha = 1.0 - level
    low = np.quantile(boot_means, alpha / 2.0)
    high = np.quantile(boot_means, 1.0 - alpha / 2.0)
    return float(low), float(high)


def holm_bonferroni(pvals: Iterable[float]) -> List[float]:
    """
    Holm-adjusted p-values.
    """
    pvals = np.asarray(list(pvals), dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted.tolist()


def run_paired_test(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    n_boot: int = N_BOOT,
    ci_level: float = CI_LEVEL,
    seed: int = RANDOM_SEED,
) -> TestResult:
    if method_a not in df.columns:
        raise ValueError(f"Column '{method_a}' not found.")
    if method_b not in df.columns:
        raise ValueError(f"Column '{method_b}' not found.")

    sub = df[[method_a, method_b]].dropna().copy()
    a = sub[method_a].to_numpy(dtype=float)
    b = sub[method_b].to_numpy(dtype=float)
    diffs = a - b

    # Wilcoxon signed-rank test on paired differences
    # zero_method='wilcox' drops zero differences
    # alternative='two-sided' matches a standard significance claim
    stat, p = wilcoxon(
        a,
        b,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    ci_low, ci_high = paired_bootstrap_ci(
        diffs,
        level=ci_level,
        n_boot=n_boot,
        seed=seed,
    )

    return TestResult(
        method_a=method_a,
        method_b=method_b,
        n=len(diffs),
        mean_a=float(np.mean(a)),
        std_a=float(np.std(a, ddof=1)),
        mean_b=float(np.mean(b)),
        std_b=float(np.std(b, ddof=1)),
        mean_diff=float(np.mean(diffs)),
        median_diff=float(np.median(diffs)),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=float(p),
        statistic=float(stat),
    )


def pretty_print(results: List[TestResult], adjust_p: bool = True) -> None:
    pvals = [r.p_value for r in results]
    adj_pvals = holm_bonferroni(pvals) if adjust_p and len(results) > 1 else pvals

    rows = []
    for r, p_adj in zip(results, adj_pvals):
        rows.append(
            {
                "comparison": f"{r.method_a} vs {r.method_b}",
                "n": r.n,
                f"mean±sd {r.method_a}": f"{r.mean_a:.2f} ± {r.std_a:.2f}",
                f"mean±sd {r.method_b}": f"{r.mean_b:.2f} ± {r.std_b:.2f}",
                "mean diff (A-B)": f"{r.mean_diff:.2f}",
                "median diff (A-B)": f"{r.median_diff:.2f}",
                f"{int(CI_LEVEL*100)}% CI of mean diff": f"[{r.ci_low:.2f}, {r.ci_high:.2f}]",
                "Wilcoxon p": f"{r.p_value:.4g}",
                "Holm-adjusted p": f"{p_adj:.4g}",
            }
        )

    out = pd.DataFrame(rows)
    print("\nPaired tests on matched instances\n")
    print(out.to_string(index=False))


def make_paper_sentence(
    result: TestResult,
    adjusted_p: float | None = None,
    split_name: str = "test",
) -> str:
    p_text = f"{result.p_value:.3g}" if adjusted_p is None else f"{adjusted_p:.3g}"
    return (
        f"On the matched {split_name} instances (N={result.n}), "
        f"{result.method_a} achieves a higher objective than {result.method_b} "
        f"(paired mean difference {result.mean_diff:.1f}, "
        f"{int(CI_LEVEL*100)}\\% CI [{result.ci_low:.1f}, {result.ci_high:.1f}], "
        f"two-sided Wilcoxon signed-rank p={p_text})."
    )


def main() -> None:
    # Default file path inside the script
    csv_path = "stats_signif_gen.csv"

    # If a file is provided in the terminal, use that instead
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]

    df = pd.read_csv(csv_path)

    results = []
    for method_a, method_b in COMPARISONS:
        results.append(run_paired_test(df, method_a, method_b))

    pretty_print(results, adjust_p=True)

    adj_pvals = (
        holm_bonferroni([r.p_value for r in results])
        if len(results) > 1
        else [r.p_value for r in results]
    )

    print("\nPaper-ready sentences:\n")
    for r, p_adj in zip(results, adj_pvals):
        print(make_paper_sentence(r, adjusted_p=p_adj, split_name="test"))

if __name__ == "__main__":
    main()