"""Moteur de backtest à poids.

Chronologie d'un jour t :
  1. le portefeuille détenu depuis la veille (poids w) réalise le rendement de t :
        r_t = Σ wᵢ·Rᵢ,t + cash·rf_t − coût_emprunt·Σ|shorts|
  2. les poids dérivent avec les prix (pas de rééquilibrage implicite)
  3. si t est une date de rééquilibrage : signal calculé avec les données
     jusqu'à la clôture de t -> on trade vers la cible au prix de clôture de t,
     frais = fee · Σ|w_cible − w_dérivé|
  4. la nouvelle cible ne gagne/perd qu'à partir de t+1 (pas de look-ahead)

Coupe-circuit (optionnel) : si la valeur du portefeuille tombe de plus de
`dd_limit` sous son plus haut, on ramène immédiatement l'exposition à
`dd_cut` × la cible (0 = tout en cash), à la clôture du jour, même hors date de
rééquilibrage. Après `dd_cooldown` jours, on revient à l'exposition normale et
le plus haut de référence repart de la valeur du jour (sinon, une fois en cash,
on ne remonterait jamais au-dessus du seuil et on resterait coupé pour toujours).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TRADING_DAYS, simple_returns
from .portfolio import PortfolioConfig, class_array, target_weights_np
from .signals import momentum_score


@dataclass
class BacktestResult:
    returns: pd.Series          # rendement net quotidien du portefeuille
    gross_returns: pd.Series    # avant frais de transaction
    costs: pd.Series            # frais de transaction payés (fraction de NAV)
    weights: pd.DataFrame       # poids DÉTENUS pendant la journée t
    cash: pd.Series             # poids cash détenu pendant t
    turnover: pd.Series         # Σ|Δw| traité à la clôture de t
    asset_returns: pd.DataFrame
    rf: pd.Series
    rebalance_dates: pd.DatetimeIndex
    breaker: pd.Series | None = None          # True les jours où le coupe-circuit est actif
    breaker_events: pd.DataFrame | None = None  # dates de coupure / reprise

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    def slice(self, start=None, end=None) -> "BacktestResult":
        """Sous-période (ex. train / test) d'un backtest calculé sur tout l'historique."""
        s = slice(start, end)
        return BacktestResult(
            self.returns.loc[s], self.gross_returns.loc[s], self.costs.loc[s],
            self.weights.loc[s], self.cash.loc[s], self.turnover.loc[s],
            self.asset_returns.loc[s], self.rf.loc[s],
            self.rebalance_dates[(self.rebalance_dates >= (start or self.returns.index[0]))
                                 & (self.rebalance_dates <= (end or self.returns.index[-1]))],
            self.breaker.loc[s] if self.breaker is not None else None,
            self.breaker_events,
        )


def rebalance_mask(index: pd.DatetimeIndex, freq: str | int) -> np.ndarray:
    """True les jours de rééquilibrage. freq : 'D', 'W', 'M' ou un entier (tous les n jours)."""
    if isinstance(freq, int):
        mask = np.zeros(len(index), dtype=bool)
        mask[::freq] = True
        return mask
    if freq == "D":
        return np.ones(len(index), dtype=bool)
    if freq not in ("W", "M"):
        raise ValueError("freq doit être 'D', 'W', 'M' ou un entier")
    period = index.to_period(freq)
    # dernier jour ouvré de chaque période présent dans les données
    return np.append(period[1:] != period[:-1], True)


def run_backtest(
    prices: pd.DataFrame,
    cfg: PortfolioConfig,
    rf: pd.Series | None = None,
    lookback: int = 126,
    skip: int = 0,
    vol_adjusted: bool = False,
    rebalance: str | int = "M",
    fee: float = 0.0005,
    borrow_cost: float = 0.0,
    scores: pd.DataFrame | None = None,
    dd_limit: float | None = None,
    dd_cut: float = 0.0,
    dd_cooldown: int = 21,
    fixed_weights: dict | None = None,
) -> BacktestResult:
    """Backtest du momentum cross-sectionnel.

    fee          : coût proportionnel par unité de notionnel traité (0.0005 = 5 bp)
    borrow_cost  : coût annuel d'emprunt des titres vendus à découvert
    scores       : signaux pré-calculés (sinon momentum_score)
    dd_limit     : seuil de drawdown du coupe-circuit (0.15 = -15 %) ; None = désactivé
    dd_cut       : exposition conservée pendant la coupure (0 = 100 % cash, 0.5 = moitié)
    dd_cooldown  : nb de jours ouvrés avant de revenir à l'exposition normale
    fixed_weights: {ticker: poids} -> allocation fixe rééquilibrée (ignore le signal) ;
                   sert aux stratégies de référence (allocation stratégique, 60/40…)
    """
    if dd_limit is not None and not (0 < dd_limit < 1):
        raise ValueError("dd_limit doit être entre 0 et 1")
    dd_cooldown = max(int(dd_cooldown), 1)
    prices = prices.sort_index().copy()
    prices.attrs = {}  # les métadonnées pandas sont recopiées à chaque opération : coûteux
    rets = simple_returns(prices)
    if rf is None:
        rf = pd.Series(0.0, index=prices.index)
    rf = rf.reindex(prices.index).ffill().fillna(0.0)
    if scores is None:
        scores = momentum_score(prices, lookback=lookback, skip=skip, vol_adjusted=vol_adjusted)
    S = scores.reindex(prices.index).values.astype(float)
    classes = class_array(list(prices.columns), cfg)
    fixed = (np.array([fixed_weights.get(c, 0.0) for c in prices.columns], dtype=float)
             if fixed_weights is not None else None)

    idx = prices.index
    T, N = rets.shape
    R = rets.values
    RF = rf.values
    borrow_daily = (1 + borrow_cost) ** (1 / TRADING_DAYS) - 1
    rebal = rebalance_mask(idx, rebalance)
    # pas de rééquilibrage tant que la covariance n'est pas estimable
    first_ok = max(lookback, cfg.cov_window)

    w = np.zeros(N)
    held = np.zeros((T, N))
    gross = np.zeros(T)
    cost = np.zeros(T)
    turn = np.zeros(T)
    rebal_dates = []
    active = np.zeros(T, dtype=bool)
    events = []
    nav = peak = 1.0
    cut, cut_until = False, -1

    for t in range(T):
        held[t] = w
        active[t] = cut
        g = 0.0
        if t > 0:
            short_notional = -w[w < 0].sum()
            g = w @ R[t] + (1 - w.sum()) * RF[t] - borrow_daily * short_notional
            gross[t] = g
            if 1 + g > 0:
                w = w * (1 + R[t]) / (1 + g)       # dérive des poids
        nav_pre = nav * (1 + g)

        tgt = None
        reenter = cut and t >= cut_until
        if reenter:                                # fin de la pause : retour à l'exposition normale
            cut, peak = False, nav_pre
            events.append((idx[t], "reprise"))
        if t >= first_ok and (rebal[t] or reenter):
            cov = np.cov(R[t - cfg.cov_window + 1: t + 1], rowvar=False).reshape(N, N)
            base = fixed if fixed is not None else target_weights_np(S[t], cov, cfg, classes)
            tgt = base * (dd_cut if cut else 1.0)
            rebal_dates.append(idx[t])
        if dd_limit is not None and not cut and t >= first_ok and nav_pre / peak - 1 <= -dd_limit:
            cut, cut_until = True, t + dd_cooldown  # déclenchement : on réduit tout de suite
            events.append((idx[t], "coupure"))
            tgt = (tgt if tgt is not None else w) * dd_cut

        if tgt is not None:
            turn[t] = np.abs(tgt - w).sum()
            cost[t] = fee * turn[t]
            w = tgt
        nav = nav * (1 + g - cost[t])
        peak = max(peak, nav)

    weights = pd.DataFrame(held, index=idx, columns=prices.columns)
    net = pd.Series(gross - cost, index=idx, name="strategy")
    return BacktestResult(
        returns=net,
        gross_returns=pd.Series(gross, index=idx, name="gross"),
        costs=pd.Series(cost, index=idx, name="costs"),
        weights=weights,
        cash=(1 - weights.sum(axis=1)).rename("cash"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        asset_returns=rets,
        rf=rf.rename("rf"),
        rebalance_dates=pd.DatetimeIndex(rebal_dates),
        breaker=pd.Series(active, index=idx, name="breaker"),
        breaker_events=pd.DataFrame(events, columns=["date", "event"]),
    )


def equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    """Benchmark : équipondéré, rééquilibré chaque jour, 100 % investi, sans frais."""
    return simple_returns(prices).mean(axis=1).rename("equal_weight")
