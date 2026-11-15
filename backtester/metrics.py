"""Métriques de performance, de risque et de diversification.
Toutes les fonctions prennent des rendements SIMPLES quotidiens."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def sharpe(returns: pd.Series, rf: pd.Series | None = None) -> float:
    ex = returns - (rf.reindex(returns.index).fillna(0) if rf is not None else 0)
    sd = ex.std()
    return float(np.sqrt(TRADING_DAYS) * ex.mean() / sd) if sd > 0 else 0.0


def performance_metrics(
    returns: pd.Series,
    rf: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
) -> dict:
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return {}
    rf_ = rf.reindex(r.index).fillna(0) if rf is not None else pd.Series(0.0, index=r.index)
    ex = r - rf_

    years = n / TRADING_DAYS
    total = float((1 + r).prod() - 1)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    downside = ex[ex < 0].std() * np.sqrt(TRADING_DAYS)
    dd = drawdown_series(r)
    max_dd = float(dd.min())

    out = {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe(r, rf_),
        "sortino": float(ex.mean() * TRADING_DAYS / downside) if downside > 0 else np.nan,
        "max_dd": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "win_rate": float((r[r != 0] > 0).mean()) if (r != 0).any() else np.nan,
    }

    if benchmark is not None:
        b = benchmark.reindex(r.index).fillna(0)
        var_b = b.var()
        beta = float(r.cov(b) / var_b) if var_b > 0 else np.nan
        out["beta"] = beta
        out["alpha_ann"] = float((ex.mean() - beta * (b - rf_).mean()) * TRADING_DAYS)
        out["correlation_bench"] = float(r.corr(b))

    if weights is not None:
        w = weights.reindex(r.index)
        out["avg_gross_exposure"] = float(w.abs().sum(axis=1).mean())
        out["avg_net_exposure"] = float(w.sum(axis=1).mean())
        out["avg_cash"] = float(1 - w.sum(axis=1).mean())
        out["time_invested"] = float((w.abs().sum(axis=1) > 1e-9).mean())

    if turnover is not None:
        out["ann_turnover"] = float(turnover.reindex(r.index).fillna(0).sum() / years)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Corrélations / diversification
# ─────────────────────────────────────────────────────────────────────────────

def average_rolling_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Corrélation moyenne entre toutes les paires d'actifs, en glissant.
    Monte typiquement vers 1 en période de crise -> la diversification disparaît."""
    n = returns.shape[1]
    if n < 2:
        return pd.Series(np.nan, index=returns.index)
    rc = returns.rolling(window).corr()
    mask = ~np.eye(n, dtype=bool)

    def avg(block):
        v = block.values
        return np.nan if np.isnan(v).all() else np.nanmean(v[mask])

    return rc.groupby(level=0).apply(avg).rename("avg_corr")


def risk_contributions(weights: pd.Series, returns_hist: pd.DataFrame) -> pd.Series:
    """Part de chaque actif dans la variance du portefeuille (somme = 1)."""
    w = weights[weights != 0]
    if w.empty:
        return pd.Series(dtype=float)
    cov = returns_hist[w.index].cov().values
    rc = w.values * (cov @ w.values)
    total = rc.sum()
    return pd.Series(rc / total if total != 0 else rc, index=w.index)


def diversification_ratio(weights: pd.Series, returns_hist: pd.DataFrame) -> float:
    """Σ|wᵢ|·σᵢ / σ_portefeuille. = 1 si aucun bénéfice de diversification."""
    w = weights[weights != 0]
    if w.empty:
        return np.nan
    cov = returns_hist[w.index].cov().values
    port = np.sqrt(w.values @ cov @ w.values)
    return float(np.abs(w.values) @ np.sqrt(np.diag(cov)) / port) if port > 0 else np.nan
