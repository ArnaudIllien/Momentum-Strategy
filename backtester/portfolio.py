"""Construction de portefeuille cross-sectionnel : classement -> sélection ->
pondération intra-jambe -> mise à l'échelle (exposition, vol target, levier).

Tout ce qui n'est pas investi reste en cash (rémunéré au taux sans risque par
le moteur). Convention : cash = 1 - Σ wᵢ (le produit des ventes à découvert
s'ajoute au cash).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .data import TRADING_DAYS

Weighting = Literal["equal", "inverse_vol", "risk_parity"]


@dataclass
class PortfolioConfig:
    # ── Sélection cross-sectionnelle ──
    n_long: int = 2                 # nb d'actifs achetés (les mieux classés)
    n_short: int = 0                # nb d'actifs vendus (les moins bien classés) ; 0 = long-only
    absolute_filter: bool = True    # long seulement si score > 0, short seulement si score < 0
                                    # -> l'emplacement non utilisé reste en cash ("dual momentum")
    # ── Pondération intra-jambe ──
    weighting: Weighting = "equal"
    cov_window: int = 63            # fenêtre d'estimation vol / covariance
    # ── Exposition ──
    long_exposure: float = 1.0      # poids total visé de la jambe longue (ex. 0.6 = 60 % investi, 40 % cash)
    short_exposure: float = 0.0     # poids total (en valeur absolue) de la jambe courte
    vol_target: float | None = None # vol annuelle cible du portefeuille (ex. 0.10) ; None = désactivé
    max_leverage: float = 1.0       # plafond sur l'exposition brute Σ|wᵢ|
    # ── Plafonds par classe d'actifs (jambe longue, en % de la valeur du portefeuille) ──
    class_caps: dict | None = None  # ex. {"Bourse": 0.6, "Obligations": 0.2, ...} ; None = désactivé
    asset_class: dict | None = None # ticker -> classe ; une classe absente de class_caps n'est pas plafonnée

    def __post_init__(self):
        if self.n_long < 0 or self.n_short < 0:
            raise ValueError("n_long / n_short doivent être >= 0")
        if self.n_short > 0 and self.short_exposure <= 0:
            raise ValueError("n_short > 0 nécessite short_exposure > 0")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage doit être > 0")
        if self.class_caps and any(not (0 <= v <= 10) for v in self.class_caps.values()):
            raise ValueError("Plafonds de classe invalides")


def apply_class_caps(w: np.ndarray, classes: np.ndarray, caps: dict, max_iter: int = 50) -> np.ndarray:
    """Plafonne le poids LONG total de chaque classe.

    L'excédent d'une classe trop chargée est redistribué aux autres actifs déjà
    retenus (au prorata de leurs poids) dont la classe a encore de la marge ;
    ce qui ne peut être placé nulle part reste en cash. Les shorts ne sont pas
    concernés. Aucun nouvel actif n'est ajouté : le signal garde la main.
    """
    w = w.copy()
    longs = w > 0
    if not longs.any():
        return w
    labels = list(dict.fromkeys(classes[longs]))
    total = {c: w[longs & (classes == c)].sum() for c in labels}
    cap = {c: caps.get(c, np.inf) for c in labels}

    pending = 0.0
    for c in labels:                       # 1. on ramène chaque classe sous son plafond
        if total[c] > cap[c] + 1e-12:
            m = longs & (classes == c)
            w[m] *= cap[c] / total[c]
            pending += total[c] - cap[c]
            total[c] = cap[c]

    for _ in range(max_iter):              # 2. on redistribue l'excédent là où il reste de la marge
        if pending <= 1e-12:
            break
        recv = [c for c in labels if cap[c] - total[c] > 1e-12]
        if not recv:
            break                          # plus de marge nulle part -> l'excédent reste en cash
        base = sum(total[c] for c in recv)
        given = 0.0
        for c in recv:
            give = min(cap[c] - total[c], pending * total[c] / base)
            m = longs & (classes == c)
            w[m] *= (total[c] + give) / total[c]
            total[c] += give
            given += give
        pending -= given
        if given <= 1e-12:
            break
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Pondérations intra-jambe (retournent des poids >= 0 qui somment à 1)
# ─────────────────────────────────────────────────────────────────────────────

def _inverse_vol(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    w = 1 / vol
    return w / w.sum()


def _risk_parity(cov: np.ndarray, n_iter: int = 1000, tol: float = 1e-12) -> np.ndarray:
    """Equal Risk Contribution : chaque actif contribue autant au risque.

    Descente par coordonnées cycliques (Griveau-Billion, Richard & Roncalli, 2013) :
    on minimise ½·wᵀΣw − (1/n)·Σ log wᵢ, problème strictement convexe dont la
    solution normalisée est l'unique portefeuille ERC long-only. Chaque mise à
    jour est la racine positive d'un trinôme, donc toujours > 0 — robuste aux
    corrélations négatives (actions / obligations), contrairement au point fixe.
    """
    n = cov.shape[0]
    cov = cov * (1 / np.mean(np.diag(cov)))   # mise à l'échelle : n'affecte pas la solution normalisée
    diag = np.clip(np.diag(cov), 1e-12, None)
    b = 1.0 / n
    w = _inverse_vol(cov)
    for _ in range(n_iter):
        w_old = w.copy()
        for i in range(n):
            c = cov[i] @ w - diag[i] * w[i]          # Σ_{j≠i} σ_ij w_j
            w[i] = (-c + np.sqrt(c * c + 4 * diag[i] * b)) / (2 * diag[i])
        if np.abs(w - w_old).max() < tol * max(w.max(), 1e-12):
            break
    return w / w.sum()


def _leg_weights(cov: np.ndarray, weighting: Weighting) -> np.ndarray:
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if weighting == "equal" or n == 1:
        return np.full(n, 1 / n)
    if weighting == "inverse_vol":
        return _inverse_vol(cov)
    if weighting == "risk_parity":
        return _risk_parity(cov)
    raise ValueError(f"weighting inconnu : {weighting}")


# ─────────────────────────────────────────────────────────────────────────────
# Poids cibles à une date
# ─────────────────────────────────────────────────────────────────────────────

def target_weights_np(scores: np.ndarray, cov: np.ndarray, cfg: PortfolioConfig,
                      classes: np.ndarray | None = None) -> np.ndarray:
    """Cœur numpy (rapide) : scores (N,) du jour + covariance quotidienne (N, N)
    estimée sur les `cov_window` derniers jours. Retourne les poids (N,) ;
    le complément à 1 est du cash."""
    n = len(scores)
    w = np.zeros(n)
    valid = np.flatnonzero(~np.isnan(scores))
    if valid.size == 0:
        return w
    ranked = valid[np.argsort(-scores[valid], kind="stable")]   # du meilleur au pire

    # -- Jambe longue
    longs = list(ranked[: cfg.n_long])
    if cfg.absolute_filter:
        longs = [i for i in longs if scores[i] > 0]
    # -- Jambe courte (sans chevauchement avec les longs)
    shorts = [i for i in ranked[::-1][: cfg.n_short] if i not in longs]
    if cfg.absolute_filter:
        shorts = [i for i in shorts if scores[i] < 0]

    for leg, n_slots, exposure, sign in ((longs, cfg.n_long, cfg.long_exposure, 1.0),
                                         (shorts, cfg.n_short, cfg.short_exposure, -1.0)):
        if not leg or n_slots == 0:
            continue
        lw = _leg_weights(cov[np.ix_(leg, leg)], cfg.weighting)
        # Emplacements vides (filtre absolu) -> la part correspondante reste en cash
        w[leg] = sign * exposure * (len(leg) / n_slots) * lw

    # -- Vol targeting : ajuste l'exposition globale à la vol cible
    if cfg.vol_target:
        act = np.flatnonzero(w)
        if act.size:
            wa = w[act]
            port_vol = float(np.sqrt(max(wa @ cov[np.ix_(act, act)] @ wa, 0.0) * TRADING_DAYS))
            if port_vol > 0:
                w *= cfg.vol_target / port_vol

    # -- Plafonds par classe d'actifs
    if cfg.class_caps and classes is not None:
        w = apply_class_caps(w, classes, cfg.class_caps)

    # -- Plafond de levier brut (ne fait que réduire : les plafonds de classe restent respectés)
    gross = np.abs(w).sum()
    if gross > cfg.max_leverage:
        w *= cfg.max_leverage / gross
    return w


def target_weights(scores: pd.Series, returns_hist: pd.DataFrame, cfg: PortfolioConfig) -> pd.Series:
    """Version pandas (pratique pour explorer une date à la main)."""
    cov = returns_hist[scores.index].iloc[-cfg.cov_window:].cov().values
    classes = class_array(list(scores.index), cfg)
    return pd.Series(target_weights_np(scores.values.astype(float), cov, cfg, classes), index=scores.index)


def class_array(tickers: list[str], cfg: PortfolioConfig) -> np.ndarray | None:
    if not cfg.class_caps:
        return None
    m = cfg.asset_class or {}
    return np.array([m.get(t, "—") for t in tickers], dtype=object)
