"""Walk-forward : ré-optimisation périodique des paramètres sur le passé seulement.

Principe, à chaque date de ré-optimisation d (tous les `step_months` mois) :
  1. on évalue chaque combinaison de paramètres sur la fenêtre d'entraînement
     [d − train_years, d[  (glissante) ou [début, d[  (ancrée)
  2. on retient la meilleure (Sharpe au-dessus du taux sans risque)
  3. on l'applique de d jusqu'à la prochaine ré-optimisation
La courbe obtenue en recollant ces segments est 100 % hors échantillon.

Astuce de calcul : chaque backtest est causal (le rendement du jour t n'utilise
que l'information jusqu'à t−1). On peut donc calculer UNE fois chaque
combinaison sur tout l'historique, puis simplement découper les segments.
Quand on change de paramètres, on paie les frais de passage d'un portefeuille
à l'autre.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from .engine import BacktestResult, run_backtest
from .metrics import sharpe
from .portfolio import PortfolioConfig

RUN_KEYS = {"lookback", "skip", "vol_adjusted", "rebalance", "fee", "borrow_cost",
            "dd_limit", "dd_cut", "dd_cooldown", "fixed_weights"}


@dataclass
class WalkForwardResult:
    returns: pd.Series            # rendements recollés hors échantillon (frais de transition inclus)
    weights: pd.DataFrame         # poids détenus
    choices: pd.DataFrame         # une ligne par ré-optimisation : paramètres retenus + Sharpe train
    train_scores: pd.DataFrame    # Sharpe train de chaque combinaison à chaque date (lignes = dates)
    switch_costs: pd.Series


def param_grid(**axes: list) -> list[dict]:
    """param_grid(lookback=[63, 126], n_long=[2, 3]) -> liste de dicts."""
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in product(*axes.values())]


def _run(prices, rf, cfg: PortfolioConfig, run_kw: dict, override: dict) -> BacktestResult:
    kw = {**run_kw, **{k: v for k, v in override.items() if k in RUN_KEYS}}
    cfg_kw = {**cfg.__dict__, **{k: v for k, v in override.items() if k not in RUN_KEYS}}
    if "lookback" in kw and kw.get("skip", 0) >= kw["lookback"]:
        kw["skip"] = kw["lookback"] - 1
    return run_backtest(prices, PortfolioConfig(**cfg_kw), rf=rf, **kw)


def walk_forward(
    prices: pd.DataFrame,
    cfg: PortfolioConfig,
    grid: list[dict],
    rf: pd.Series | None = None,
    run_kw: dict | None = None,
    train_years: float = 3.0,
    step_months: int = 6,
    anchored: bool = False,
    progress: Callable[[float], None] | None = None,
) -> WalkForwardResult:
    run_kw = dict(run_kw or {})
    fee = run_kw.get("fee", 0.0005)
    rf = rf if rf is not None else pd.Series(0.0, index=prices.index)

    # 1. Un backtest complet par combinaison
    runs = []
    for i, combo in enumerate(grid):
        runs.append(_run(prices, rf, cfg, run_kw, combo))
        if progress:
            progress((i + 1) / len(grid))
    idx = prices.index

    # 2. Dates de ré-optimisation (premier jour ouvré de chaque période)
    live_start = max(r.rebalance_dates[0] for r in runs if len(r.rebalance_dates))
    first = live_start + pd.DateOffset(months=int(round(train_years * 12)))
    if first >= idx[-1]:
        raise ValueError("Historique trop court pour la fenêtre d'entraînement choisie")
    targets = pd.date_range(first, idx[-1], freq=pd.DateOffset(months=step_months))
    reopt = pd.DatetimeIndex(sorted({idx[idx.searchsorted(d)] for d in targets if d <= idx[-1]}))

    # 3. Choix à chaque date, sur le passé uniquement
    labels = [", ".join(f"{k}={v}" for k, v in c.items()) for c in grid]
    scores, chosen = [], []
    for d in reopt:
        t0 = live_start if anchored else d - pd.DateOffset(months=int(round(train_years * 12)))
        win = slice(max(t0, live_start), d - pd.Timedelta(days=1))
        s = [sharpe(r.returns.loc[win], rf) for r in runs]
        scores.append(s)
        chosen.append(int(np.nanargmax(s)))

    # 4. Recollage des segments + frais de passage d'un jeu de paramètres à l'autre
    ret = pd.Series(np.nan, index=idx, name="walk_forward")
    W = pd.DataFrame(np.nan, index=idx, columns=prices.columns)
    switch = pd.Series(0.0, index=idx, name="switch_costs")
    bounds = list(reopt) + [None]
    prev = None
    for k, d in enumerate(reopt):
        seg = slice(d, bounds[k + 1] - pd.Timedelta(days=1) if bounds[k + 1] is not None else None)
        r = runs[chosen[k]]
        ret.loc[seg] = r.returns.loc[seg]
        W.loc[seg] = r.weights.loc[seg].values
        if prev is not None and prev != chosen[k]:
            # Le jour d, on détient le portefeuille du nouveau jeu au lieu de l'ancien
            turn = float(np.abs(r.weights.loc[d] - runs[prev].weights.loc[d]).sum())
            switch.loc[d] = fee * turn
            ret.loc[d] -= fee * turn
        prev = chosen[k]

    choices = pd.DataFrame([{**grid[c], "train_sharpe": scores[k][c]} for k, c in enumerate(chosen)],
                           index=reopt)
    choices.index.name = "date"
    return WalkForwardResult(
        returns=ret.dropna(),
        weights=W.dropna(how="all"),
        choices=choices,
        train_scores=pd.DataFrame(scores, index=reopt, columns=labels),
        switch_costs=switch.loc[reopt[0]:],
    )
