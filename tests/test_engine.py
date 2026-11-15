"""Tests hors-ligne (données synthétiques). Lancer : python -m tests.test_engine"""

import numpy as np
import pandas as pd

from backtester import PortfolioConfig, run_backtest, risk_contributions
from backtester.data import align_business_days, constant_risk_free
from backtester.engine import rebalance_mask
from backtester.portfolio import _risk_parity

IDX = pd.bdate_range("2020-01-01", periods=400)


def make_prices(seed=0, n=4, drift=None):
    rng = np.random.default_rng(seed)
    drift = drift if drift is not None else np.linspace(-0.0005, 0.001, n)
    vols = np.linspace(0.005, 0.03, n)
    r = rng.normal(drift, vols, size=(len(IDX), n))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=IDX, columns=list("ABCD")[:n])


def test_no_lookahead():
    p = make_prices()
    cfg = PortfolioConfig(n_long=2, n_short=1, short_exposure=0.5, weighting="inverse_vol")
    base = run_backtest(p, cfg, lookback=60, rebalance="W")
    cut = IDX[250]
    p2 = p.copy()
    p2.loc[p2.index > cut] *= np.linspace(1, 3, (p2.index > cut).sum())[:, None]  # choc futur
    shocked = run_backtest(p2, cfg, lookback=60, rebalance="W")
    pd.testing.assert_series_equal(base.returns.loc[:cut], shocked.returns.loc[:cut])


def test_partial_investment_earns_cash_rate():
    p = make_prices(n=1, drift=[0.002])  # tendance haussière -> score > 0
    rf = constant_risk_free(IDX, 0.04)
    cfg = PortfolioConfig(n_long=1, long_exposure=0.5, cov_window=20)
    res = run_backtest(p, cfg, rf=rf, lookback=20, rebalance="D", fee=0.0)
    r_a = p.pct_change()["A"]
    live = res.weights["A"] > 0
    expected = 0.5 * r_a[live] + 0.5 * rf[live]
    assert np.allclose(res.returns[live], expected), "50 % actif + 50 % cash attendu"


def test_absolute_filter_goes_to_cash():
    p = make_prices(n=3, drift=[-0.01] * 3)  # tout baisse -> aucun long autorisé
    rf = constant_risk_free(IDX, 0.03)
    cfg = PortfolioConfig(n_long=2, absolute_filter=True, cov_window=20)
    res = run_backtest(p, cfg, rf=rf, lookback=60, rebalance="D", fee=0.0)
    late = res.returns.index[100:]
    assert (res.weights.loc[late].abs().sum(axis=1) < 1e-12).mean() > 0.95
    assert np.allclose(res.returns.loc[late][res.cash.loc[late] == 1], rf.loc[late][res.cash.loc[late] == 1])


def test_transaction_costs():
    p = make_prices(n=1, drift=[0.002])
    cfg = PortfolioConfig(n_long=1, cov_window=20)
    res = run_backtest(p, cfg, lookback=20, rebalance="M", fee=0.001)
    first = res.rebalance_dates[0]
    assert np.isclose(res.turnover[first], 1.0) and np.isclose(res.costs[first], 0.001)


def test_vol_target_and_leverage_cap():
    p = make_prices()
    cfg = PortfolioConfig(n_long=2, n_short=2, short_exposure=1.0, absolute_filter=False,
                          vol_target=0.05, max_leverage=1.5)
    res = run_backtest(p, cfg, lookback=60, rebalance="W")
    assert res.weights.abs().sum(axis=1).max() <= 1.5 + 0.05  # + petite dérive intra-semaine


def test_risk_parity_equal_contributions():
    p = make_prices()
    r = p.pct_change().dropna()
    w = pd.Series(_risk_parity(r.cov().values), index=r.columns)
    rc = risk_contributions(w, r)
    assert np.allclose(rc, 1 / len(rc), atol=1e-6)


def test_risk_parity_negative_correlations():
    """Cas réel qui faisait diverger l'ancien algorithme : actions / obligations
    négativement corrélées + un actif très volatil."""
    vols = np.array([0.18, 0.20, 0.07, 0.06, 0.37])
    corr = np.array([[1.0, 0.9, -0.3, -0.2, 0.3],
                     [0.9, 1.0, -0.3, -0.2, 0.3],
                     [-0.3, -0.3, 1.0, 0.9, -0.1],
                     [-0.2, -0.2, 0.9, 1.0, -0.1],
                     [0.3, 0.3, -0.1, -0.1, 1.0]])
    cov = np.outer(vols, vols) * corr / 252
    w = _risk_parity(cov)
    rc = w * (cov @ w)
    assert (w > 0).all() and np.isclose(w.sum(), 1)
    assert np.allclose(rc / rc.sum(), 0.2, atol=1e-6)
    assert w[4] < 0.15 and w[2] + w[3] > 0.4     # le plus volatil a le moins de poids


def test_calendar():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")  # inclut des week-ends
    df = pd.DataFrame({"BTC": range(14), "SPX": [1.0] * 14}, index=idx, dtype=float)
    df.loc[df.index.dayofweek >= 5, "SPX"] = np.nan
    out = align_business_days(df)
    assert (out.index.dayofweek < 5).all() and not out.isna().any().any()
    m = rebalance_mask(pd.bdate_range("2024-01-01", "2024-03-31"), "M")
    assert m.sum() == 3


def test_walk_forward_single_combo_equals_backtest():
    from backtester.walkforward import walk_forward
    p = make_prices()
    cfg = PortfolioConfig(n_long=2)
    wf = walk_forward(p, cfg, [{"lookback": 40}], run_kw={"rebalance": "W"}, train_years=0.5, step_months=2)
    bt = run_backtest(p, cfg, lookback=40, rebalance="W")
    pd.testing.assert_series_equal(wf.returns, bt.returns.loc[wf.returns.index], check_names=False)
    assert wf.switch_costs.sum() == 0


def test_walk_forward_no_lookahead():
    from backtester.walkforward import walk_forward, param_grid
    idx = pd.bdate_range("2018-01-01", periods=900)
    rng = np.random.default_rng(1)
    p = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (900, 4)), axis=0), index=idx, columns=list("ABCD"))
    grid = param_grid(lookback=[20, 60, 120], n_long=[1, 2])
    kw = dict(run_kw={"rebalance": "W"}, train_years=1, step_months=3)
    base = walk_forward(p, PortfolioConfig(), grid, **kw)
    cut = idx[600]
    p2 = p.copy()
    p2.loc[p2.index > cut, "A"] *= np.linspace(1, 5, (p2.index > cut).sum())  # A explose dans le futur
    shocked = walk_forward(p2, PortfolioConfig(), grid, **kw)
    pd.testing.assert_series_equal(base.returns.loc[:cut], shocked.returns.loc[:cut])
    pd.testing.assert_frame_equal(base.choices.loc[:cut], shocked.choices.loc[:cut])


def _crash_prices():
    """A monte, puis chute de 40 % en 20 jours, puis remonte. B et C baissent doucement."""
    n = len(IDX)
    a = np.r_[np.full(200, 0.003), np.full(20, -0.025), np.full(n - 220, 0.002)]
    bc = np.full((n, 2), -0.0005)
    r = np.column_stack([a, bc])
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=IDX, columns=list("ABC"))


def test_circuit_breaker_cuts_and_reenters():
    p = _crash_prices()
    rf = constant_risk_free(IDX, 0.03)
    # sans filtre de tendance : à la reprise on doit être réinvesti même si tout baisse
    cfg = PortfolioConfig(n_long=1, cov_window=20, absolute_filter=False)
    kw = dict(rf=rf, lookback=20, rebalance="M", fee=0.0)
    res = run_backtest(p, cfg, dd_limit=0.10, dd_cut=0.0, dd_cooldown=15, **kw)
    ev = res.breaker_events
    assert list(ev["event"][:2]) == ["coupure", "reprise"]
    t_cut = IDX.get_loc(ev["date"].iloc[0])
    t_back = IDX.get_loc(ev["date"].iloc[1])
    assert t_back - t_cut == 15
    # Drawdown au déclenchement : juste au-delà de -10 % (pas avant)
    dd = (1 + res.returns).cumprod()
    dd = dd / dd.cummax() - 1
    assert dd.iloc[t_cut] <= -0.10 < dd.iloc[t_cut - 1]
    # Pendant la coupure : 100 % cash, rendement = taux sans risque
    seg = slice(IDX[t_cut + 1], IDX[t_back])
    assert np.allclose(res.weights.loc[seg].abs().sum(axis=1), 0)
    assert np.allclose(res.returns.loc[seg], rf.loc[seg])
    assert res.breaker.loc[seg].all()
    # Après la reprise : réinvesti
    assert res.weights.iloc[t_back + 1].sum() > 0.99
    # Le coupe-circuit limite la perte par rapport à la version sans
    base = run_backtest(p, cfg, **kw)
    assert drawdown_min(res.returns) > drawdown_min(base.returns)


def drawdown_min(r):
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def test_circuit_breaker_off_changes_nothing():
    p = make_prices()
    cfg = PortfolioConfig(n_long=2, weighting="risk_parity")
    a = run_backtest(p, cfg, lookback=60, rebalance="W")
    b = run_backtest(p, cfg, lookback=60, rebalance="W", dd_limit=0.99)  # seuil jamais atteint
    pd.testing.assert_series_equal(a.returns, b.returns)
    assert not a.breaker.any() and a.breaker_events.empty


def test_class_caps_redistribute_then_cash():
    from backtester.portfolio import apply_class_caps
    cls = np.array(["Bourse", "Bourse", "Oblig"], dtype=object)
    caps = {"Bourse": 0.6, "Oblig": 0.5}
    # Bourse 80 % -> 60 % ; les 20 % libérés vont aux obligations (marge 30 %)
    w = apply_class_caps(np.array([0.4, 0.4, 0.2]), cls, caps)
    assert np.allclose(w, [0.3, 0.3, 0.4])
    # Toutes les classes saturées -> l'excédent reste en cash
    w = apply_class_caps(np.array([0.4, 0.3, 0.3]), cls, {"Bourse": 0.6, "Oblig": 0.2})
    assert np.isclose(w[:2].sum(), 0.6) and np.isclose(w[2], 0.2) and np.isclose(w.sum(), 0.8)
    # Classe non plafonnée : absorbe l'excédent ; shorts inchangés
    w = apply_class_caps(np.array([0.5, 0.3, -0.2]), np.array(["Bourse", "Autre", "Bourse"], dtype=object),
                         {"Bourse": 0.25})
    assert np.isclose(w[0], 0.25) and np.isclose(w[1], 0.55) and np.isclose(w[2], -0.2)


def test_class_caps_in_backtest():
    p = make_prices()
    classes = {"A": "Bourse", "B": "Bourse", "C": "Bourse", "D": "Oblig"}
    cfg = PortfolioConfig(n_long=3, absolute_filter=False, class_caps={"Bourse": 0.5, "Oblig": 0.3},
                          asset_class=classes)
    res = run_backtest(p, cfg, lookback=60, rebalance="M", fee=0.0)
    for d in res.rebalance_dates[:-1]:
        w = res.weights.iloc[IDX.get_loc(d) + 1]          # poids juste après le rééquilibrage
        assert w[["A", "B", "C"]].sum() <= 0.5 + 1e-9 and w["D"] <= 0.3 + 1e-9


def test_comparison_strategies():
    from backtester.strategies import compare_strategies, strategic_weights
    p = make_prices()
    buckets = {"A": "Bourse", "B": "Bourse", "C": "Obligations", "D": "Matières premières"}
    targets = {"Bourse": 0.6, "Obligations": 0.2, "Matières premières": 0.1, "Crypto / Autres": 0.1}
    sw = strategic_weights(list(p.columns), buckets, targets)
    assert np.isclose(sw["A"], 0.3) and np.isclose(sum(sw.values()), 0.9)  # pas de crypto -> 10 % cash
    cfg = PortfolioConfig(n_long=2, class_caps=targets, asset_class=buckets)
    out = compare_strategies(p, cfg, constant_risk_free(IDX, 0.02), {"lookback": 60, "rebalance": "M"},
                             buckets, targets)
    assert set(out) == {"xs_mom", "xs_mom_nocap", "ts_mom", "strategic", "equal", "risk_parity"}
    first = out["strategic"].rebalance_dates[0]
    w = out["strategic"].weights.iloc[IDX.get_loc(first) + 1]
    assert np.allclose(w.values, [sw[t] for t in p.columns])
    # même date de départ pour toutes les stratégies
    assert len({r.rebalance_dates[0] for r in out.values()}) == 1


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(tests)} tests passés")
