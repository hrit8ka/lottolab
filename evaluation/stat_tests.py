"""Direct tests of 'is there any signal at all?' -- run these BEFORE any model.
Monte Carlo p-values avoid relying on asymptotic approximations."""
from __future__ import annotations
import numpy as np

N, K = 40, 6


def _simulate_fair(T: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.random((T, N))
    idx = np.argpartition(r, K, axis=1)[:, :K]
    X = np.zeros((T, N), dtype=np.int8)
    np.put_along_axis(X, idx, 1, axis=1)
    return X


def _chi2_numbers(X):
    T = len(X)
    exp = T * K / N
    return float((((X.sum(axis=0) - exp) ** 2) / exp).sum())


def _pair_stat(X):
    T = len(X)
    H = X.astype(np.int32)
    C = H.T @ H
    iu = np.triu_indices(N, 1)
    exp = T * (K * (K - 1)) / (N * (N - 1))
    return float((((C[iu] - exp) ** 2) / exp).sum())


def _repeat_stat(X):
    """Mean overlap between consecutive draws (expected 0.9 if independent)."""
    return float((X[1:] & X[:-1]).sum(axis=1).mean())


def mc_pvalue(stat_fn, X, n_sim=1000, seed=0, two_sided=False):
    rng = np.random.default_rng(seed)
    obs = stat_fn(X)
    sims = np.array([stat_fn(_simulate_fair(len(X), rng)) for _ in range(n_sim)])
    if two_sided:
        centre = sims.mean()
        p = (np.sum(np.abs(sims - centre) >= abs(obs - centre)) + 1) / (n_sim + 1)
    else:
        p = (np.sum(sims >= obs) + 1) / (n_sim + 1)
    return obs, float(p)


def run_all(X: np.ndarray, n_sim: int = 1000, seed: int = 0) -> dict:
    out = {}
    out["number_uniformity_chi2"] = mc_pvalue(_chi2_numbers, X, n_sim, seed)
    out["pair_independence"] = mc_pvalue(_pair_stat, X, n_sim, seed + 1)
    out["consecutive_draw_overlap"] = mc_pvalue(_repeat_stat, X, n_sim, seed + 2, two_sided=True)
    return out
