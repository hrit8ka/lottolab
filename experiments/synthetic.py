"""Pipeline validation on synthetic lotteries.
  Env A: fair lottery            -> every agent should look like random.
  Env B: biased lottery          -> do any agents detect a real bias?

python -m experiments.synthetic [--draws 1500] [--bias 1.5] [--seeds 5]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from agents.strategies import default_agents
from evaluation.metrics import HIT_MEAN, holm, min_detectable_edge, summarize_hits
from evaluation.walk_forward import walk_forward

N, K = 40, 6


def simulate(T: int, weights: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    w = np.ones(N) if weights is None else weights
    g = np.log(w) + rng.gumbel(size=(T, N))   # weighted sampling w/o replacement
    idx = np.argpartition(-g, K, axis=1)[:, :K]
    X = np.zeros((T, N), dtype=np.int8)
    np.put_along_axis(X, idx, 1, axis=1)
    return X


def biased_weights(bias: float, n_boosted: int = 6) -> np.ndarray:
    w = np.ones(N)
    w[:n_boosted] = bias
    return w


def run_env(X, label, seed):
    res = walk_forward(X, default_agents(), warmup=100, seed=seed)
    rows, pvals = [], {}
    for name in res.columns:
        s = summarize_hits(res[name].to_numpy())
        pvals[name] = s["p_value"]
        rows.append({"env": label, "agent": name, **s})
    adj = holm(pvals)
    df = pd.DataFrame(rows)
    df["p_holm"] = df["agent"].map(adj)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--bias", type=float, default=1.5, help="weight multiplier on 6 boosted numbers")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print(f"Draws={args.draws}; min detectable edge (80% power) = {min_detectable_edge(args.draws - 100):.3f} hits")
    frames = []
    for s in range(args.seeds):
        rng = np.random.default_rng(1000 + s)
        frames.append(run_env(simulate(args.draws, None, rng), "A_fair", s).assign(seed=s))
        frames.append(run_env(simulate(args.draws, biased_weights(args.bias), rng), "B_biased", s).assign(seed=s))
    df = pd.concat(frames)
    agg = (df.groupby(["env", "agent"])
             .agg(mean_hits=("mean_hits", "mean"), edge=("edge_vs_theory", "mean"),
                  min_p_holm=("p_holm", "min"), sig_runs=("p_holm", lambda x: int((x < 0.05).sum())))
             .round(4))
    print(f"\nTheory mean hits = {HIT_MEAN}\n")
    print(agg)
    out = "experiments/results/synthetic_summary.csv"
    agg.to_csv(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
