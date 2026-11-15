"""Chargement et alignement des données.

Convention : calendrier des JOURS OUVRÉS (lun–ven). Les actifs qui cotent le
week-end (crypto) sont ramenés sur ce calendrier : le rendement du lundi
inclut donc le mouvement du week-end (prix de clôture vendredi -> lundi).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def align_business_days(prices: pd.DataFrame, max_ffill: int = 3) -> pd.DataFrame:
    """Ramène un panel de prix sur les jours ouvrés communs.

    - supprime samedi/dimanche
    - forward-fill limité (jours fériés propres à une place : ex. Thanksgiving
      pour le S&P alors que l'EUR/USD cote) -> rendement nul ce jour-là
    - supprime les dates où un actif n'a encore aucun historique
    """
    prices = prices.sort_index()
    prices = prices[prices.index.dayofweek < 5]
    # Prix <= 0 (ex. pétrole WTI à -37 $ en avril 2020) : rendement non défini
    prices = prices.where(prices > 0)
    prices = prices.ffill(limit=max_ffill)
    return prices.dropna(how="any")


def _download_close(tickers: list[str], start, end) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:  # un seul ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def get_currency(ticker: str) -> str:
    """Devise de cotation (appel réseau). 'USD' par défaut si inconnue."""
    try:
        import yfinance as yf

        return yf.Ticker(ticker).fast_info.get("currency") or "USD"
    except Exception:
        return "USD"


def convert_to_usd(close: pd.DataFrame, currencies: dict[str, str], start, end) -> pd.DataFrame:
    """Convertit en USD les colonnes cotées dans une autre devise.
    Gère les cotations en centimes (GBp / GBX à Londres, ILA, ZAc)."""
    subunits = {"GBp": ("GBP", 100), "GBX": ("GBP", 100), "ILA": ("ILS", 100), "ZAc": ("ZAR", 100)}
    todo = {}
    for t, ccy in currencies.items():
        if t not in close.columns or not ccy or ccy == "USD":
            continue
        base, div = subunits.get(ccy, (ccy.upper(), 1))
        todo[t] = (base, div)
    if not todo:
        return close
    pairs = sorted({f"{b}USD=X" for b, _ in todo.values()})
    fx = _download_close(pairs, start, end)
    close = close.copy()
    for t, (base, div) in todo.items():
        pair = f"{base}USD=X"
        if pair not in fx.columns or fx[pair].isna().all():
            raise ValueError(f"Taux de change {pair} introuvable pour convertir {t}")
        rate = fx[pair].reindex(close.index).ffill().bfill()
        close[t] = close[t] / div * rate
    return close


def search_assets(query: str, max_results: int = 8) -> list[dict]:
    """Recherche Yahoo Finance par nom : [{symbol, name, type, exchange}]."""
    import yfinance as yf

    try:
        quotes = yf.Search(query, max_results=max_results, news_count=0).quotes
    except Exception:
        return []
    return [{"symbol": q["symbol"], "name": q.get("shortname") or q.get("longname") or q["symbol"],
             "type": q.get("typeDisp") or q.get("quoteType", ""), "exchange": q.get("exchDisp", "")}
            for q in quotes if q.get("symbol")]


def load_prices(tickers: list[str], start, end, currencies: dict[str, str] | None = None) -> pd.DataFrame:
    """Télécharge les clôtures ajustées (yfinance) -> DataFrame (dates x tickers), en USD
    si `currencies` indique des devises de cotation."""
    close = _download_close(tickers, start, end)
    if close.empty:
        raise ValueError(f"Aucune donnée pour {tickers}. Yahoo Finance limite parfois temporairement les "
                         "requêtes : réessayez dans quelques minutes.")
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing:
        raise ValueError(f"Tickers sans données : {missing}")
    close = close[tickers]
    if currencies:
        close = convert_to_usd(close, currencies, start, end)
    first_dates = {t: close[t].first_valid_index() for t in tickers}
    out = align_business_days(close)
    out.attrs["first_dates"] = first_dates  # pour signaler l'actif qui limite l'historique
    return out


def load_risk_free(index: pd.DatetimeIndex, start, end, ticker: str = "^IRX") -> pd.Series:
    """Taux sans risque QUOTIDIEN aligné sur `index`.

    ^IRX = rendement annualisé du T-bill 13 semaines, en %.
    Conversion : (1 + y)^(1/252) - 1. Retourne 0 si indisponible.
    """
    try:
        import yfinance as yf

        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        y = raw["Close"]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y.index = pd.to_datetime(y.index).tz_localize(None)
        y = y.reindex(index).ffill().bfill() / 100.0
        return ((1 + y) ** (1 / TRADING_DAYS) - 1).rename("rf")
    except Exception:
        return pd.Series(0.0, index=index, name="rf")


def constant_risk_free(index: pd.DatetimeIndex, annual_rate: float) -> pd.Series:
    """Taux sans risque constant (utile pour tests / hors-ligne)."""
    daily = (1 + annual_rate) ** (1 / TRADING_DAYS) - 1
    return pd.Series(daily, index=index, name="rf")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements SIMPLES : seuls ceux-ci s'agrègent entre actifs (Σ wᵢ·rᵢ)."""
    return prices.pct_change().fillna(0.0)
