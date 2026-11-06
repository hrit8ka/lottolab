"""Writes data/sample/synthetic_draws.csv -- FAKE data from a fair simulated lottery,
used only so the app and tests run before real draws are ingested."""
import numpy as np, pandas as pd
from experiments.synthetic import simulate

T = 1200
X = simulate(T, None, np.random.default_rng(42))
dates = pd.date_range(end="2026-11-04", periods=T, freq="W-WED")  # placeholder cadence
rows = []
for d, x in zip(dates, X):
    n = np.flatnonzero(x) + 1
    rows.append([d.date().isoformat(), *n])
pd.DataFrame(rows, columns=["draw_date", *[f"n{i}" for i in range(1, 7)]]).to_csv("data/sample/synthetic_draws.csv", index=False)
print("wrote data/sample/synthetic_draws.csv (SYNTHETIC)")
