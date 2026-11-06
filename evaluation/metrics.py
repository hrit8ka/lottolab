from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.special import comb

N, K = 40, 6


def hypergeom_pmf(k: int) -> float:
    return comb(K, k) * comb(N - K, K - k) / comb(N, K)


HIT_PMF = np.array([hypergeom_pmf(k) for k in range(K + 1)])
HIT_MEAN = K * K / N                                   # 0.9
HIT_VAR = K * (K / N) * (1 - K / N) * (N - K) / (N - 1)  # ~0.667
HIT_SD = HIT_VAR ** 0.5                                # ~0.816


def summarize_hits(hits: np.ndarray) -> dict:
    n = len(hits)
    mean = hits.mean()
    se = hits.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_1samp(hits, HIT_MEAN)
    return {
        "n_draws": n,
        "mean_hits": mean,
        "se": se,
        "edge_vs_theory": mean - HIT_MEAN,
        "p_value": p,
        "rate_3plus": float((hits >= 3).mean()),
        "theory_3plus": float(HIT_PMF[3:].sum()),
    }


def hit_distribution(hits: np.ndarray) -> np.ndarray:
    return np.bincount(hits.astype(int), minlength=K + 1) / len(hits)


def min_detectable_edge(n_draws: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true edge in mean hits detectable with the given draws (two-sided z-test)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * HIT_SD / np.sqrt(n_draws))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj
