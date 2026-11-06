"""Chronological walk-forward backtest. At step t each agent sees only draws[:t]."""
from __future__ import annotations
import numpy as np
import pandas as pd


def walk_forward(X: np.ndarray, agents, warmup: int = 100, seed: int = 0,
                 start: int | None = None) -> pd.DataFrame:
    """Returns DataFrame (index = draw index) of hits per agent. `start` lets you
    reserve an untouched holdout: only draws >= start are scored (default warmup)."""
    rng = np.random.default_rng(seed)
    start = warmup if start is None else start
    T = len(X)
    out = {a.name: np.zeros(T - start, dtype=np.int8) for a in agents}
    for i, t in enumerate(range(start, T)):
        hist = X[:t]  # strictly past
        actual = X[t]
        for a in agents:
            picks = a.select(hist, rng)
            out[a.name][i] = actual[np.array(picks) - 1].sum()
    df = pd.DataFrame(out)
    df.index = np.arange(start, T)
    return df
