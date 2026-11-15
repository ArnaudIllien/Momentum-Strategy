"""Stratégies de comparaison, toutes calculées avec le même moteur, les mêmes
frais, le même calendrier de rééquilibrage et la même date de départ que le
momentum cross-sectionnel — seule la règle d'allocation change.

Les variantes « momentum » reprennent aussi les overlays de l'utilisateur
(plafonds de classe, coupe-circuit). Les références passives n'utilisent ni
signal ni coupe-circuit.
"""

from __future__ import annotations

import pandas as pd

from .engine import BacktestResult, run_backtest
from .portfolio import PortfolioConfig
from .signals import momentum_score

STRATEGY_KEYS = ["xs_mom", "xs_mom_nocap", "ts_mom", "strategic", "equal", "risk_parity"]


def strategic_weights(tickers: list[str], buckets: dict[str, str], targets: dict[str, float]) -> dict:
    """Allocation stratégique : chaque grande classe reçoit sa cible, répartie à parts
    égales entre ses actifs. Une classe sans actif sélectionné laisse sa part en cash."""
    out = {}
    for b, tgt in targets.items():
        members = [t for t in tickers if buckets.get(t) == b]
        for t in members:
            out[t] = tgt / len(members)
    return out


def compare_strategies(
    prices: pd.DataFrame,
    cfg: PortfolioConfig,
    rf: pd.Series,
    run_kw: dict,
    buckets: dict[str, str],
    targets: dict[str, float],
    main: BacktestResult | None = None,
) -> dict[str, BacktestResult]:
    n = prices.shape[1]
    cols = list(prices.columns)
    passive_kw = {k: v for k, v in run_kw.items() if not k.startswith("dd_")}
    ones = pd.DataFrame(1.0, index=prices.index, columns=cols)   # « signal » neutre
    base = {**cfg.__dict__}

    out: dict[str, BacktestResult] = {}
    # 1. Momentum cross-sectionnel (configuration de la barre latérale)
    out["xs_mom"] = main if main is not None else run_backtest(prices, cfg, rf=rf, **run_kw)
    # 2. Même chose sans plafonds de classe -> effet des plafonds
    if cfg.class_caps:
        out["xs_mom_nocap"] = run_backtest(prices, PortfolioConfig(**{**base, "class_caps": None}), rf=rf, **run_kw)
    # 3. Momentum time-series : chaque actif est acheté si SA propre tendance est positive (1/N), sinon cash
    ts_scores = momentum_score(prices, lookback=run_kw.get("lookback", 126), skip=run_kw.get("skip", 0))
    ts_cfg = PortfolioConfig(**{**base, "n_long": n, "n_short": 0, "short_exposure": 0.0,
                                "absolute_filter": True, "weighting": "equal"})
    out["ts_mom"] = run_backtest(prices, ts_cfg, rf=rf, scores=ts_scores, **run_kw)
    # 4. Allocation stratégique fixe par grande classe (ex. 60/20/10/10)
    out["strategic"] = run_backtest(prices, PortfolioConfig(n_long=n), rf=rf,
                                    fixed_weights=strategic_weights(cols, buckets, targets), **passive_kw)
    # 5. Équipondéré rééquilibré (avec frais)
    out["equal"] = run_backtest(prices, PortfolioConfig(n_long=n), rf=rf,
                                fixed_weights={t: 1 / n for t in cols}, **passive_kw)
    # 6. Parité des risques sur tout l'univers, sans signal
    rp_cfg = PortfolioConfig(n_long=n, absolute_filter=False, weighting="risk_parity",
                             cov_window=cfg.cov_window)
    out["risk_parity"] = run_backtest(prices, rp_cfg, rf=rf, scores=ones, **passive_kw)
    return out
