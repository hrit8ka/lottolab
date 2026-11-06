"""Leak-free features. Every function takes `history`: a (T, 40) 0/1 matrix of
PAST draws only, and describes the state just before the next draw."""
from __future__ import annotations
import numpy as np
import pandas as pd

N = 40
K = 6
WINDOWS = (10, 25, 50, 100)


def draws_to_matrix(draws: pd.DataFrame) -> np.ndarray:
    """DataFrame with n1..n6 -> (T, 40) binary matrix, in chronological order."""
    nums = draws[[f"n{i}" for i in range(1, 7)]].to_numpy()
    X = np.zeros((len(nums), N), dtype=np.int8)
    rows = np.repeat(np.arange(len(nums)), K)
    X[rows, nums.ravel() - 1] = 1
    return X


def number_features(history: np.ndarray) -> pd.DataFrame:
    """One row per number (1..40) with frequency, recency and deviation features."""
    T = len(history)
    feats = {"number": np.arange(1, N + 1)}
    for w in WINDOWS:
        feats[f"freq_{w}"] = history[-w:].sum(axis=0) if T else np.zeros(N)
    feats["freq_all"] = history.sum(axis=0) if T else np.zeros(N)
    if T:
        # draws since last appearance (0 = appeared in the latest draw)
        recency = np.where(history.any(axis=0), np.argmax(history[::-1], axis=0), T)
    else:
        recency = np.zeros(N)
    feats["recency"] = recency
    expected_recent = 0.15 * min(T, 50)
    feats["dev_50_vs_expected"] = feats["freq_50"] - expected_recent
    df = pd.DataFrame(feats)
    return df


def pair_cooccurrence(history: np.ndarray) -> np.ndarray:
    """(40, 40) count of draws in which both numbers appeared."""
    H = history.astype(np.int32)
    C = H.T @ H
    np.fill_diagonal(C, 0)
    return C


def draw_level_features(numbers: tuple[int, ...]) -> dict:
    n = sorted(numbers)
    return {
        "sum": sum(n),
        "odd": sum(x % 2 for x in n),
        "low": sum(x <= 20 for x in n),
        "consecutive_pairs": sum(b - a == 1 for a, b in zip(n, n[1:])),
        "max_gap": max(b - a for a, b in zip(n, n[1:])),
        "birthday_range": sum(x <= 31 for x in n),
    }
