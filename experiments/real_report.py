"""One-shot analysis of the real draws.   python -m experiments.real_report [--n-sim 2000]

Protocol (fixed in advance, so results can't be cherry-picked):
  * Data-quality checks first.
  * Direct tests: number uniformity, pair independence, draw-to-draw overlap (Monte Carlo p-values).
  * Walk-forward backtest, warm-up 100 draws, every later draw scored once.
  * Primary metric: mean hits per draw vs the theoretical 0.9. Family = the 4 non-random strategies,
    Holm-corrected. The random agent is a sanity check (should look like theory).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agents.strategies import default_agents
from evaluation import stat_tests
from evaluation.metrics import HIT_MEAN, holm, min_detectable_edge, summarize_hits
from evaluation.walk_forward import walk_forward
from features.features import draws_to_matrix
from ingestion import storage

SEEDS = 5


def section(t):
    return f"\n{'=' * 8} {t} {'=' * 8}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(storage.DB_PATH))
    ap.add_argument("--n-sim", type=int, default=2000)
    args = ap.parse_args()

    df = storage.load_draws(Path(args.db))
    X = draws_to_matrix(df)
    out = []
    p = out.append

    p(section("DATA QUALITY"))
    p(f"draws: {len(df)}   range: {df.draw_date.min().date()} -> {df.draw_date.max().date()}")
    wd = df.draw_date.dt.day_name().value_counts()
    p("weekday counts: " + ", ".join(f"{k}={v}" for k, v in wd.items()))
    gaps = df.draw_date.diff().dt.days
    big = df.loc[gaps > 14, "draw_date"].dt.date.astype(str).tolist()
    p(f"gaps > 14 days before: {big if big else 'none'}")
    same = int((X[1:] == X[:-1]).all(axis=1).sum())
    p(f"identical consecutive draws: {same}")
    p(f"draws per era: Saturday-only era (< 2018-09) = {(df.draw_date < '2018-09-01').sum()}, "
      f"twice-weekly era = {(df.draw_date >= '2018-09-01').sum()}")

    p(section("NUMBER FREQUENCIES (descriptive)"))
    cnt = X.sum(axis=0)
    exp = len(X) * 0.15
    order = np.argsort(-cnt)
    p(f"expected per number if fair: {exp:.1f}  (SD about {np.sqrt(len(X) * 0.15 * 0.85):.1f})")
    p("most drawn : " + ", ".join(f"{i + 1}({cnt[i]})" for i in order[:5]))
    p("least drawn: " + ", ".join(f"{i + 1}({cnt[i]})" for i in order[-5:]))

    p(section("DIRECT TESTS OF RANDOMNESS (Monte Carlo, n_sim=%d)" % args.n_sim))
    res = stat_tests.run_all(X, n_sim=args.n_sim)
    adj = holm({k: v[1] for k, v in res.items()})
    for k, (stat, pv) in res.items():
        p(f"{k:28s} statistic={stat:10.3f}  p={pv:.4f}  p_holm={adj[k]:.4f}")
    p("(small p = unlikely under a fair lottery; large p = no evidence of bias)")

    p(section("WALK-FORWARD BACKTEST"))
    frames = [walk_forward(X, default_agents(), warmup=100, seed=s) for s in range(SEEDS)]
    n_scored = len(frames[0])
    p(f"scored draws: {n_scored}   theory mean hits: {HIT_MEAN}   "
      f"min detectable edge (80% power): {min_detectable_edge(n_scored):.3f} hits")
    rows, pv = [], {}
    for name in frames[0].columns:
        s = summarize_hits(frames[0][name].to_numpy())
        s["mean_all_seeds"] = float(np.mean([fr[name].mean() for fr in frames]))
        rows.append({"agent": name, **s})
        if name != "random":
            pv[name] = s["p_value"]
    ad = holm(pv)
    tab = pd.DataFrame(rows)
    tab["p_holm"] = tab["agent"].map(ad)
    cols = ["agent", "mean_hits", "mean_all_seeds", "edge_vs_theory", "se", "p_value", "p_holm", "rate_3plus", "theory_3plus"]
    p(tab[cols].round(4).to_string(index=False))
    p("p_holm is blank for 'random' (control, not part of the tested family).")

    sig = [a for a, v in ad.items() if v < 0.05]
    p(section("READING THE RESULT"))
    if sig:
        p(f"Holm-significant strategies: {sig}. Treat as a lead to re-test on future draws, not a proven edge.")
    else:
        p("No strategy beats chance after correction. This is the expected outcome for a fair lottery.")

    text = "\n".join(out)
    print(text)
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/real_report.txt").write_text(text)
    print("\nSaved experiments/results/real_report.txt")


if __name__ == "__main__":
    main()
