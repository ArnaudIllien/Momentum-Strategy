"""Signaux. Toutes les fonctions n'utilisent que l'information disponible à la
clôture de t (pas de look-ahead) ; le décalage d'exécution est géré par le moteur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS


def momentum_score(
    prices: pd.DataFrame,
    lookback: int = 126,
    skip: int = 0,
    vol_adjusted: bool = False,
    vol_window: int = 63,
) -> pd.DataFrame:
    """Momentum = P[t-skip] / P[t-lookback] - 1.

    - `skip` : ignore les derniers jours (classique 12-1 mois : lookback=252,
      skip=21) pour éviter l'effet de retournement court terme.
    - `vol_adjusted` : divise par la vol réalisée -> compare les actifs à risque
      égal (sinon BTC domine toujours le classement en valeur absolue).
    """
    if skip >= lookback:
        raise ValueError("skip doit être < lookback")
    score = prices.shift(skip) / prices.shift(lookback) - 1
    if vol_adjusted:
        vol = prices.pct_change().rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
        score = score / vol
    return score


def rolling_volatility(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Vol annualisée glissante."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
