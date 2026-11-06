import numpy as np
import pandas as pd
import pytest

from agents.strategies import default_agents
from evaluation.metrics import HIT_MEAN, HIT_PMF, HIT_SD, holm, min_detectable_edge
from evaluation.walk_forward import walk_forward
from experiments.synthetic import simulate
from features.features import draws_to_matrix, number_features
from ingestion.sources import rows_from_dataframe, rows_from_html
from ingestion.validator import InvalidDraw, validate_draw, validate_numbers


def test_combinatorics():
    assert abs(HIT_PMF.sum() - 1) < 1e-12
    assert HIT_MEAN == pytest.approx(0.9)
    assert (HIT_PMF * np.arange(7)).sum() == pytest.approx(0.9)
    assert HIT_SD == pytest.approx(0.8165, abs=1e-3)


@pytest.mark.parametrize("bad", [[1, 2, 3, 4, 5], [1, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 41]])
def test_validator_rejects(bad):
    with pytest.raises(InvalidDraw):
        validate_numbers(bad)


def test_validator_accepts_and_sorts():
    d, n = validate_draw("04/11/2026", [40, 1, 7, 22, 13, 30])
    assert n == (1, 7, 13, 22, 30, 40) and d.year == 2026


def test_csv_parser_tolerant():
    df = pd.DataFrame({"Draw Date": ["2026-01-03", "2026-01-07"], "Numbers": ["1 2 3 4 5 6", "7, 8, 9, 10, 11, 99"]})
    good, bad = rows_from_dataframe(df)
    assert len(good) == 1 and len(bad) == 1


def test_html_parser():
    html = "<table><tr><td>03/01/2026</td><td>1</td><td>5</td><td>9</td><td>14</td><td>22</td><td>40</td></tr></table>"
    good, bad = rows_from_html(html)
    assert good[0][1] == (1, 5, 9, 14, 22, 40)


def test_agents_return_valid_combos():
    X = simulate(300, None, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    for a in default_agents():
        for hist in (X[:0], X[:5], X[:250]):
            p = a.select(hist, rng)
            assert len(p) == 6 and len(set(p)) == 6 and min(p) >= 1 and max(p) <= 40


def test_features_no_lookahead():
    X = simulate(200, None, np.random.default_rng(0))
    f1 = number_features(X[:150])
    X2 = X.copy(); X2[150:] = 0  # destroy the future
    f2 = number_features(X2[:150])
    pd.testing.assert_frame_equal(f1, f2)


def test_walk_forward_no_lookahead():
    """Changing the target draw must not change what agents pick for it."""
    X = simulate(220, None, np.random.default_rng(3))
    agents = default_agents()
    r1 = walk_forward(X, agents, warmup=100, seed=7)
    X2 = X.copy(); X2[-1] = X2[-2]  # alter only the last (revealed-after) draw
    r2 = walk_forward(X2, default_agents(), warmup=100, seed=7)
    # all steps before the last one are identical
    pd.testing.assert_frame_equal(r1.iloc[:-1], r2.iloc[:-1])


def test_random_agent_matches_theory():
    X = simulate(3000, None, np.random.default_rng(5))
    res = walk_forward(X, [default_agents()[0]], warmup=100, seed=1)
    assert abs(res["random"].mean() - 0.9) < 0.06


def test_holm_monotone():
    adj = holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adj["a"] <= adj["b"] <= adj["c"] <= 1


def test_power_scale():
    assert min_detectable_edge(1400) == pytest.approx(0.0611, abs=2e-3)


def test_yelo_parser_keeps_only_loto():
    from ingestion.yelo import parse_yelo_html
    def row(label, nums, jp):
        tds = "<br>".join(str(n) for n in nums)
        return f"<tr><td>{label}</td><td>{tds}</td><td>{jp}</td></tr>"
    html = "<table>" + "".join([
        row("04 November, 2026 - Wednesday<br><a>Loto Result</a>", [1, 6, 17, 26, 35, 39], "10 Millions"),
        row("04 November, 2026 - Wednesday<br><a>Loto Plus Result</a>", [2, 5, 9, 13, 23, 26], "5 Millions"),
        row("03 November, 2026 - Tuesday<br><a>Loto Vert Result</a>", [5, 9, 13, 15, 17, 21, 27], "-"),
        row("31 October, 2026 - Saturday<br><a>Loto Result</a>", [4, 5, 11, 17, 20, 23], "30 Millions"),
    ]) + "</table>"
    good, bad = parse_yelo_html(html)
    assert [g[1] for g in good] == [(1, 6, 17, 26, 35, 39), (4, 5, 11, 17, 20, 23)]
    assert str(good[0][0]) == "2026-11-04" and not bad
