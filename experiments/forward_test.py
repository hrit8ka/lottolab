"""Forward test: log each strategy's picks BEFORE a draw, score them AFTER.
This can't be cherry-picked: picks are generated once per (draw date, strategy), seeded by the date,
and the log refuses to overwrite or to accept picks for a draw already in the database.

  python -m experiments.forward_test picks --date 2026-11-07     # before the draw
  python -m experiments.forward_test score                       # after: update data first
(after a draw: python -m ingestion.yelo && python -m ingestion.run_pipeline)
"""
from __future__ import annotations
import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agents.strategies import default_agents
from evaluation.metrics import HIT_MEAN
from features.features import draws_to_matrix
from ingestion import storage

LOG = Path("data/forward_log.csv")
COLS = ["logged_at_utc", "target_date", "agent", "pick_no", "n1", "n2", "n3", "n4", "n5", "n6", "data_through"]


def read_log(log: Path = LOG) -> pd.DataFrame:
    if not Path(log).exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(log, dtype={"target_date": str, "data_through": str})


def make_picks(db: Path, target: date, per_agent: int = 1, log: Path = LOG) -> pd.DataFrame:
    df = storage.load_draws(Path(db))
    last = df.draw_date.max().date()
    if last >= target:
        raise SystemExit(f"Database already has a draw on/after {target} (latest: {last}). "
                         "Picks must be logged BEFORE the draw.")
    X = draws_to_matrix(df)
    existing = read_log(log)
    done = set(zip(existing.target_date, existing.agent))
    rows = []
    for i, a in enumerate(default_agents()):
        if (target.isoformat(), a.name) in done:
            print(f"  {a.name}: already logged for {target}, keeping the original picks")
            continue
        rng = np.random.default_rng(int(target.strftime("%Y%m%d")) + 1000 * i)
        for k in range(per_agent):
            p = a.select(X, rng)
            rows.append({"logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "target_date": target.isoformat(), "agent": a.name, "pick_no": k + 1,
                         **{f"n{j + 1}": v for j, v in enumerate(p)}, "data_through": last.isoformat()})
    new = pd.DataFrame(rows, columns=COLS)
    if len(new):
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        pd.concat([existing, new]).to_csv(log, index=False)
    return pd.concat([existing, new])[lambda d: d.target_date == target.isoformat()]


def score_log(db: Path, log: Path = LOG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (detail, summary). Only picks whose draw is now in the database are scored."""
    lg = read_log(log)
    if lg.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = storage.load_draws(Path(db))
    actual = {d.date().isoformat(): set(r) for d, r in
              zip(df.draw_date, df[[f"n{i}" for i in range(1, 7)]].to_numpy().tolist())}
    lg = lg[lg.target_date.isin(actual)].copy()
    if lg.empty:
        return pd.DataFrame(), pd.DataFrame()
    lg["picks"] = lg[[f"n{i}" for i in range(1, 7)]].apply(lambda r: " ".join(f"{int(x):02d}" for x in r), axis=1)
    lg["hits"] = [len(set(int(x) for x in r[[f"n{i}" for i in range(1, 7)]]) & actual[r.target_date])
                  for _, r in lg.iterrows()]
    lg["winning"] = lg.target_date.map(lambda d: " ".join(f"{x:02d}" for x in sorted(actual[d])))
    detail = lg[["target_date", "agent", "pick_no", "picks", "winning", "hits"]].sort_values(["target_date", "agent"])
    summ = (detail.groupby("agent").hits.agg(picks="count", mean_hits="mean", total_hits="sum").reset_index())
    summ["draws_scored"] = detail.target_date.nunique()
    summ["expected_mean"] = HIT_MEAN
    summ["edge"] = summ.mean_hits - HIT_MEAN
    return detail, summ


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("picks")
    p1.add_argument("--date", required=True, help="draw date, YYYY-MM-DD")
    p1.add_argument("--per-agent", type=int, default=1)
    p1.add_argument("--db", default=str(storage.DB_PATH))
    p2 = sub.add_parser("score")
    p2.add_argument("--db", default=str(storage.DB_PATH))
    a = ap.parse_args()

    if a.cmd == "picks":
        target = datetime.strptime(a.date, "%Y-%m-%d").date()
        if target.weekday() not in (2, 5):
            print(f"Note: {target} is a {target.strftime('%A')}; regular Loto draws are Wednesday and Saturday.")
        out = make_picks(Path(a.db), target, a.per_agent)
        print(f"\nPicks logged for {target} (data through {out.data_through.iloc[0]}):\n")
        for _, r in out.iterrows():
            print(f"  {r.agent:15s} #{int(r.pick_no)}  " + " - ".join(f"{int(r[f'n{i}']):02d}" for i in range(1, 7)))
        print("\nEvery combination has exactly the same chance as any other in a fair lottery.")
    else:
        detail, summ = score_log(Path(a.db))
        if detail.empty:
            print("Nothing to score yet: no logged draw date is in the database.")
            return
        print(detail.to_string(index=False))
        print("\n" + summ.round(3).to_string(index=False))
        detail.to_csv("data/forward_scored.csv", index=False)
        print("\nSaved data/forward_scored.csv. With only a few draws these numbers are mostly noise.")


if __name__ == "__main__":
    main()
