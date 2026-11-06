"""Loto Lab dashboard.  Run:  streamlit run dashboard/app.py"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents.strategies import ThompsonAgent, default_agents
from evaluation import stat_tests
from evaluation.metrics import HIT_PMF, HIT_MEAN, holm, hit_distribution, min_detectable_edge, summarize_hits
from evaluation.walk_forward import walk_forward
from features.features import draw_level_features, draws_to_matrix, number_features, pair_cooccurrence
from ingestion import storage
from experiments.forward_test import LOG, read_log, score_log

st.set_page_config(page_title="Loto Lab", page_icon="🎲", layout="wide")


@st.cache_data
def load_data():
    db = storage.DB_PATH
    if db.exists():
        df = storage.load_draws(db)
        if len(df) >= 150:
            return df, False
    sample = pd.read_csv("data/sample/synthetic_draws.csv", parse_dates=["draw_date"])
    return sample, True


@st.cache_data
def run_backtest(n_rows: int, seeds: int, holdout: int):
    df, _ = load_data()
    X = draws_to_matrix(df)
    frames = [walk_forward(X, default_agents(), warmup=100, seed=s) for s in range(seeds)]
    return frames, len(X)


df, is_synthetic = load_data()
X = draws_to_matrix(df)

st.title("🎲 Loto Lab — Mauritius Loto 6/40")
if is_synthetic:
    st.warning("Showing **synthetic demo data** (fair simulated lottery). Run the ingestion pipeline to load real draws.")
st.caption("Research tool. Each draw is independent in a fair lottery, so no method here can raise your chance of matching numbers. "
           "The scoreboard exists to show that honestly.")

page = st.sidebar.radio("Page", ["Dashboard", "Number explorer", "Prediction lab", "Scoreboard", "Forward test"])
st.sidebar.metric("Draws loaded", len(df))
st.sidebar.write(f"{df.draw_date.min().date()} → {df.draw_date.max().date()}")

if page == "Dashboard":
    c1, c2, c3 = st.columns(3)
    c1.metric("Possible combinations", f"{3_838_380:,}")
    c2.metric("Expected matches (random pick)", f"{HIT_MEAN:.2f}")
    c3.metric("Draws analysed", len(df))
    f = number_features(X)
    fig = px.bar(f, x="number", y="freq_all", title="Appearances per number (all draws)")
    fig.add_hline(y=len(X) * 0.15, line_dash="dash", annotation_text="expected if fair")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Is anything non-random?")
    n_sim = st.slider("Monte Carlo simulations", 200, 3000, 500, 100)
    if st.button("Run statistical tests"):
        res = stat_tests.run_all(X, n_sim=n_sim)
        out = pd.DataFrame([{"test": k, "statistic": v[0], "p_value": v[1]} for k, v in res.items()])
        adj = holm({r.test: r.p_value for r in out.itertuples()})
        out["p_holm"] = out["test"].map(adj)
        st.dataframe(out, hide_index=True)
        st.caption("p_holm is Holm-adjusted across the 3 tests. A small p-value means 'unlikely under a fair lottery'; large p-values mean no evidence of bias.")

elif page == "Number explorer":
    f = number_features(X)
    window = st.selectbox("Frequency window", [10, 25, 50, 100], index=2)
    exp = 0.15 * window
    fig = px.bar(f, x="number", y=f"freq_{window}", title=f"Last {window} draws (expected {exp:.1f})")
    fig.add_hline(y=exp, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.bar(f, x="number", y="recency", title="Draws since last appearance"), use_container_width=True)
    C = pair_cooccurrence(X)
    st.plotly_chart(px.imshow(C, x=list(range(1, 41)), y=list(range(1, 41)), title="Pair co-occurrence counts",
                              color_continuous_scale="Blues"), use_container_width=True)

elif page == "Prediction lab":
    st.subheader("Generate combinations")
    st.info("These are model-selected combinations, not forecasts. In a fair lottery every combination has "
            "the same 1-in-3,838,380 chance of winning the jackpot.")
    agents = {a.name: a for a in default_agents() if a.name != "random"} | {"random": default_agents()[0]}
    name = st.selectbox("Strategy", list(agents), index=list(agents).index("bayes_thompson"))
    n_combo = st.slider("How many combinations", 1, 10, 5)
    seed = st.number_input("Seed (for reproducibility)", 0, 10_000, 0)
    agent = agents[name]
    st.write(agent.description)
    if st.button("Generate"):
        rng = np.random.default_rng(int(seed))
        rows = []
        for i in range(n_combo):
            p = agent.select(X, rng)
            d = draw_level_features(tuple(p))
            rows.append({"#": i + 1, "numbers": " - ".join(f"{x:02d}" for x in p), "sum": d["sum"],
                         "odd/even": f"{d['odd']}/{6 - d['odd']}", "numbers ≤31": d["birthday_range"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if name == "bayes_thompson":
            pm = ThompsonAgent().posterior_mean(X)
            st.caption(f"Posterior appearance rates range {pm.min():.3f}–{pm.max():.3f} (a fair lottery is exactly 0.150 for every number).")

elif page == "Scoreboard":
    st.subheader("Walk-forward backtest (each pick uses only earlier draws)")
    seeds = st.slider("Random seeds", 1, 5, 2)
    frames, T = run_backtest(len(X), seeds, 0)
    rows, pv = [], {}
    for name in frames[0].columns:
        allhits = np.concatenate([fr[name].to_numpy() for fr in frames])
        s = summarize_hits(frames[0][name].to_numpy())
        s["mean_hits(all seeds)"] = allhits.mean()
        pv[name] = s["p_value"]
        rows.append({"agent": name, **s})
    adj = holm(pv)
    tab = pd.DataFrame(rows)
    tab["p_holm"] = tab["agent"].map(adj)
    st.dataframe(tab[["agent", "n_draws", "mean_hits", "edge_vs_theory", "p_value", "p_holm", "rate_3plus"]].round(4), hide_index=True)
    n_scored = len(frames[0])
    st.caption(f"Theory: mean {HIT_MEAN:.2f} hits. With {n_scored} scored draws the smallest edge detectable at 80% power is "
               f"≈{min_detectable_edge(n_scored):.3f} hits. Results within that band are indistinguishable from chance. "
               "p_holm corrects for comparing several strategies.")
    dist = pd.DataFrame({"hits": range(7), "theory": HIT_PMF})
    for name in frames[0].columns:
        dist[name] = hit_distribution(frames[0][name].to_numpy())
    st.plotly_chart(px.bar(dist.melt("hits", var_name="who", value_name="share"), x="hits", y="share",
                           color="who", barmode="group", title="Hit distribution vs theory"), use_container_width=True)
    cum = pd.DataFrame({n: (frames[0][n] - HIT_MEAN).cumsum() for n in frames[0].columns})
    st.plotly_chart(px.line(cum, title="Cumulative hits above expectation (a real edge would drift upward)"), use_container_width=True)

elif page == "Forward test":
    st.subheader("Forward test: picks logged before each draw, scored after")
    st.caption("Generate picks from a terminal so they are timestamped and cannot be re-rolled: "
               "`python -m experiments.forward_test picks --date YYYY-MM-DD`. After the draw run the Yelo download, "
               "the ingestion pipeline, then reload this page.")
    lg = read_log()
    if lg.empty:
        st.info("No picks logged yet.")
    else:
        st.write("Logged picks")
        st.dataframe(lg, hide_index=True)
        detail, summ = score_log(storage.DB_PATH)
        if detail.empty:
            st.info("None of the logged draws are in the database yet.")
        else:
            st.write("Scored")
            st.dataframe(detail, hide_index=True)
            st.dataframe(summ.round(3), hide_index=True)
            st.caption(f"Expected mean hits per pick: {HIT_MEAN:.2f}. A handful of draws says nothing about skill.")
