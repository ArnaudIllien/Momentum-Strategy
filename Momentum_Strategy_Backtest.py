"""Quant Strategy Dashboard — momentum cross-sectionnel multi-actifs.

Lancer : streamlit run Momentum_Strategy_Backtest.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester import PortfolioConfig, load_prices, load_risk_free, performance_metrics, run_backtest
from backtester.engine import equal_weight_benchmark
from backtester.walkforward import param_grid, walk_forward
from backtester.metrics import (
    average_rolling_correlation, diversification_ratio, drawdown_series, risk_contributions, sharpe,
)
from backtester.data import TRADING_DAYS
from backtester.report import build_pdf
from backtester.data import get_currency, search_assets
from backtester.universe import (BUCKETS, DEFAULT_CAPS, DEFAULT_SELECTION, TICKER_TO_CLASS, UNIVERSE, bucket,
                                 label, register)
from backtester.strategies import STRATEGY_KEYS, compare_strategies

st.set_page_config(page_title="Quant Strategy Dashboard", page_icon="📈", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# STYLE DES GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

# Palette catégorielle validée (ordre fixe = sécurité daltonisme) ; gris neutre pour cash / "Autres"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#898781"
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]  # bleu ↔ gris ↔ rouge
C_STRAT, C_BENCH = SERIES[0], SERIES[1]


def style(fig: go.Figure, title: str, y_fmt: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5, zeroline=True, zerolinewidth=1)
    if y_fmt:
        fig.update_yaxes(tickformat=y_fmt)
    return fig


REPORT: dict = {"figs": {}}   # figures / tableaux réutilisés par le rapport PDF


def show(fig: go.Figure, key: str | None = None):
    st.plotly_chart(fig, width="stretch")
    if key:
        REPORT["figs"][key] = fig


# ══════════════════════════════════════════════════════════════════════════════
# TRADUCTIONS
# ══════════════════════════════════════════════════════════════════════════════

LANG = {
    "fr": {
        "title": "📈 Quant Strategy Dashboard — Momentum cross-sectionnel",
        "expander_explain": "💡 Comment fonctionne cette stratégie ?",
        "expander_limits": "⚠️ Limites du modèle",
        "universe": "Univers d'actifs", "search_title": "🔎 Ajouter n'importe quel actif",
        "search": "Rechercher (nom ou ticker)", "search_pick": "Actifs ajoutés",
        "search_ph": "Choisir dans les résultats ou taper un ticker",
        "converted": "convertis en USD",
        "period": "Période", "date_start": "Début", "date_end": "Fin", "train_pct": "Part train (%)",
        "signal": "Signal", "lookback": "Lookback momentum (jours)", "skip": "Jours récents ignorés (skip)",
        "vol_adj": "Momentum ajusté du risque (÷ volatilité)",
        "portfolio": "Portefeuille", "n_long": "Nb d'actifs achetés", "allow_short": "Autoriser la vente à découvert",
        "n_short": "Nb d'actifs vendus", "short_exp": "Exposition short", "abs_filter":
        "Filtre de tendance (long seulement si momentum > 0, sinon cash)",
        "weighting": "Pondération", "long_exp": "Part investie (jambe longue)",
        "use_vt": "Cibler une volatilité", "vt": "Volatilité cible (annuelle)", "max_lev": "Levier brut max",
        "execution": "Exécution", "rebal": "Rééquilibrage", "fee": "Frais (bp par unité traitée)",
        "borrow": "Coût d'emprunt des shorts (%/an)",
        "w_equal": "Poids égaux", "w_inverse_vol": "Inverse de la volatilité", "w_risk_parity": "Parité des risques",
        "f_D": "Quotidien", "f_W": "Hebdomadaire", "f_M": "Mensuel",
        "err_assets": "Sélectionne au moins 3 actifs.", "err_dates": "La date de début doit précéder la date de fin.",
        "err_slots": "Nb achetés + nb vendus doit être ≤ au nombre d'actifs.",
        "loading": "Téléchargement des données…", "computing": "Backtest en cours…",
        "effective": "Période effective", "limited_by": "début limité par l'historique de",
        "tab_perf": "Performance", "tab_alloc": "Allocation & cash", "tab_corr": "Corrélations & risque",
        "tab_assets": "Par actif", "tab_robust": "Robustesse", "tab_data": "Données",
        "strategy": "Stratégie", "bench": "Équipondéré (100 % investi)", "cash_curve": "Cash (taux sans risque)",
        "train": "Train", "test": "Test", "log_scale": "Échelle logarithmique",
        "equity": "Valeur du portefeuille (base 1)", "drawdown": "Drawdown", "roll_sharpe": "Sharpe glissant 6 mois",
        "headline": "Hors échantillon (test)",
        "m_cagr": "Rendement annualisé", "m_vol": "Volatilité", "m_sharpe": "Sharpe", "m_sortino": "Sortino",
        "m_maxdd": "Max drawdown", "m_calmar": "Calmar", "m_win": "Jours gagnants", "m_beta": "Bêta vs équipondéré",
        "m_alpha": "Alpha annualisé", "m_corr": "Corrélation vs équipondéré", "m_gross": "Exposition brute moy.",
        "m_net": "Exposition nette moy.", "m_cash": "Cash moyen", "m_inv": "Temps investi",
        "m_turn": "Rotation annuelle", "m_total": "Rendement total",
        "weights_title": "Poids détenus dans le temps (au-dessus de 0 : long, en dessous : short)",
        "others": "Autres", "cash": "Cash",
        "exposure_title": "Exposition et cash", "gross": "Exposition brute", "net": "Exposition nette",
        "current_title": "Poids cibles au dernier rééquilibrage", "costs_paid": "Frais payés (total)",
        "corr_title": "Matrice de corrélation des rendements quotidiens", "corr_period": "Période",
        "full": "Tout l'historique", "avg_corr_title": "Corrélation moyenne entre actifs (glissante 3 mois)",
        "rc_title": "Contribution au risque du portefeuille actuel", "div_ratio": "Ratio de diversification",
        "div_help": "Σ|wᵢ|·σᵢ / σ du portefeuille. 1 = aucun bénéfice de diversification.",
        "assets_table": "Statistiques par actif (sur toute la période)",
        "a_class": "Classe", "a_cagr": "Rdt annualisé", "a_vol": "Volatilité", "a_sharpe": "Sharpe",
        "a_dd": "Max DD", "a_w": "Poids moyen", "a_long": "% jours long", "a_short": "% jours short",
        "a_contrib": "Contribution au P&L",
        "robust_intro": "Grille lookback × nombre d'actifs achetés. On choisit le meilleur couple **sur le train "
                        "uniquement**, puis on regarde ce qu'il donne sur le test. Un bon paramètre se trouve sur "
                        "un plateau de valeurs proches, pas sur un pic isolé.",
        "robust_btn": "Lancer la grille", "grid_train": "Sharpe — train", "grid_test": "Sharpe — test",
        "best_train": "Meilleur couple (train)", "its_test": "Son Sharpe sur le test",
        "fee_title": "Sensibilité aux frais (Sharpe sur le test)", "fee_axis": "Frais (bp)",
        "yes": "Oui", "no": "Non",
        "pdf_title": "📄 Rapport PDF",
        "pdf_caption": "Reprend les paramètres, les indicateurs et les graphiques de cette page (et le "
                       "walk-forward s'il a été lancé). Génération : ~10 s.",
        "pdf_btn": "Générer le rapport PDF", "pdf_spinner": "Export des graphiques et mise en page…",
        "pdf_dl": "📥 Télécharger le PDF", "pdf_stale": "Les paramètres ont changé depuis la génération : "
        "régénère le rapport.", "pdf_error": "Impossible de générer le PDF",
        "pdf_doc_title": "Rapport de backtest — Momentum cross-sectionnel", "pdf_assets_n": "actifs",
        "pdf_params": "Paramètres", "pdf_universe": "Univers", "pdf_results": "Résultats",
        "pdf_summary": "Sur la période de test, la stratégie affiche un Sharpe de {s:.2f} contre {b:.2f} pour "
                       "le portefeuille équipondéré, pour un rendement annualisé de {cagr:.1%}, un drawdown "
                       "maximal de {dd:.1%} et {cash:.0%} de cash en moyenne.",
        "tab_cmp": "Comparaison",
        "cmp_short_cols": ["Rdt ann.", "Vol.", "Sharpe", "Sortino", "Max DD", "Calmar", "Cash moy.", "Rotation"],
        "caps_header": "Plafonds par classe d'actifs", "caps_on": "Activer les plafonds",
        "caps_help": "Poids long maximal de chaque grande classe, en % du portefeuille. L'excédent est "
                     "redistribué aux autres actifs retenus qui ont encore de la marge, sinon il reste en cash.",
        "caps_reassign": "Classe des actifs ajoutés par la recherche",
        "caps_sum_note": "Somme des plafonds = {tot:.0%} : au-delà, le reste est forcément en cash.",
        "b_Bourse": "Bourse", "b_Obligations": "Obligations", "b_Matières premières": "Matières premières",
        "b_Crypto / Autres": "Crypto / Autres",
        "class_title": "Exposition longue par classe d'actifs",
        "s_xs_mom": "Momentum cross-sectionnel", "s_xs_mom_nocap": "Momentum CS sans plafonds",
        "s_ts_mom": "Momentum time-series", "s_strategic": "Allocation stratégique {alloc}",
        "s_equal": "Équipondéré", "s_risk_parity": "Parité des risques",
        "cmp_intro": "Mêmes actifs, mêmes frais, même fréquence de rééquilibrage et même date de départ : seule "
                     "la règle d'allocation change.\n"
                     "- **Momentum time-series** : chaque actif est acheté (1/N) si *sa propre* tendance est "
                     "positive, sinon sa part reste en cash.\n"
                     "- **Allocation stratégique** : les plafonds deviennent des poids fixes par classe, répartis à "
                     "parts égales entre les actifs de la classe.\n"
                     "- **Parité des risques** : chaque actif contribue autant au risque, sans aucun signal.",
        "cmp_intro_pdf": "Mêmes actifs, frais, rééquilibrage et date de départ ; seule la règle d'allocation "
                         "change. Les variantes momentum reprennent les plafonds et le coupe-circuit ; les "
                         "références passives non.",
        "cmp_equity": "Valeur des stratégies (base 1)", "cmp_scatter": "Rendement vs volatilité",
        "cmp_corr": "Corrélation entre stratégies (rendements quotidiens)",
        "cmp_note": "Les variantes momentum reprennent les plafonds et le coupe-circuit de la barre latérale ; "
                    "les références passives (stratégique, équipondéré, parité des risques) n'en utilisent pas. "
                    "Une classe sans actif sélectionné laisse sa part de l'allocation stratégique en cash.",
        "dd_header": "Coupe-circuit", "dd_on": "Activer le coupe-circuit sur drawdown",
        "dd_help": "Si le portefeuille perd plus que le seuil depuis son plus haut, on réduit immédiatement "
                   "l'exposition, puis on revient à la normale après la pause.",
        "dd_limit": "Seuil de drawdown (0.15 = −15 %)", "dd_cut": "Exposition conservée pendant la coupure",
        "dd_cooldown": "Pause avant reprise (jours ouvrés)", "dd_triggers": "Déclenchements",
        "dd_time": "Du temps en coupure", "dd_shade_note": "zones rouges sur les graphiques",
        "dd_line": "seuil du coupe-circuit",
        "tab_wf": "Walk-forward",
        "wf_intro": "À chaque date de ré-optimisation, on choisit le couple (lookback, nb d'actifs) qui a eu le "
                    "meilleur Sharpe sur la fenêtre d'entraînement **passée**, puis on l'applique jusqu'à la "
                    "date suivante. La courbe recollée est entièrement hors échantillon : c'est l'estimation la "
                    "plus honnête de ce qu'aurait donné la méthode « j'optimise mes paramètres régulièrement ».",
        "wf_train": "Fenêtre d'entraînement (années)", "wf_step": "Ré-optimisation tous les", "months": "mois",
        "wf_anchored": "Fenêtre ancrée", "wf_anchored_help": "Ancrée : on s'entraîne depuis le début de "
                       "l'historique. Glissante : seulement sur les N dernières années.",
        "wf_nl_range": "Nb d'actifs achetés testés (min – max)", "wf_btn": "Lancer le walk-forward",
        "wf_vs_fixed": "vs fixes", "wf_changes": "Changements de paramètres", "wf_switch": "Frais de transition",
        "wf_fixed": "Paramètres fixes (barre latérale)",
        "wf_equity": "Walk-forward vs paramètres fixes vs équipondéré",
        "wf_chosen": "Paramètre retenu",
        "wf_promise": "Sharpe promis (fenêtre d'entraînement) vs obtenu (période suivante)",
        "wf_promised": "Promis (train)", "wf_realised": "Obtenu (période suivante)",
        "wf_promise_caption": "En moyenne : promis {promised:.2f}, obtenu {realised:.2f}. Un écart important "
                              "signifie que la meilleure combinaison du passé doit beaucoup au hasard.",
        "dl_returns": "📥 Rendements quotidiens (CSV)", "dl_weights": "📥 Poids détenus (CSV)",
        "dl_prices": "📥 Prix alignés (CSV)",
        "explain_md": """
### L'idée : classer les actifs entre eux

Le momentum **cross-sectionnel** ne demande pas « cet actif monte-t-il ? » mais « **lequel monte le plus
par rapport aux autres** ? ». À chaque rééquilibrage :

| Étape | Ce qui se passe |
|---|---|
| 1 | Score de chaque actif = performance sur les `lookback` derniers jours, en ignorant les `skip` derniers (effet de retournement court terme) |
| 2 | Option : score ÷ volatilité, pour comparer les actifs à risque égal (sinon le plus volatil gagne souvent) |
| 3 | On **achète les N premiers** et, si autorisé, on **vend à découvert les M derniers** |
| 4 | On répartit chaque jambe : poids égaux, inverse de la volatilité, ou parité des risques |
| 5 | On ajuste l'exposition : part investie, filtre de tendance, volatilité cible, levier max |
| 6 | Ce qui n'est pas investi reste en **cash rémunéré au taux sans risque** (T-bill 3 mois US) |

Les positions décidées à la clôture du jour t ne rapportent qu'à partir de t+1, et les frais sont payés sur ce
qui est réellement échangé.

### Ne pas être investi à 100 %
- **Part investie** < 100 % : le reste est en cash.
- **Filtre de tendance** : on n'achète un actif classé dans le top que si son momentum est positif. Si tout baisse, on finit en cash.
- **Volatilité cible** : l'exposition diminue quand les marchés s'agitent.
- **Coupe-circuit** : si la perte depuis le plus haut dépasse un seuil, on coupe (tout ou partie) immédiatement, puis on revient après une pause.

### Lire les résultats
- **Sharpe** (au-dessus du taux sans risque) : > 1 bon, 0–1 acceptable, < 0 ne bat pas le cash.
- **Équipondéré** : référence naïve toujours investie à 100 %. Une stratégie qui ne la bat pas en Sharpe n'apporte rien.
- **Train / Test** : les paramètres se règlent sur le train ; le test, jamais vu, dit si ça tient.
""",
        "limits_md": """
1. **Exécution à la clôture** : le signal et le trade se font au même prix de clôture — optimiste.
2. **Frais linéaires** : pas de spread variable ni d'impact de marché.
3. **Calendrier jours ouvrés** : pour le Bitcoin, le rendement du lundi inclut le week-end.
4. **Futures en continu** (`=F`) : les prix Yahoo ignorent le coût/gain de roulement des contrats.
5. **Biais de survie** : l'univers est choisi aujourd'hui, avec des actifs qui ont survécu.
6. **Short idéal** : coût d'emprunt constant, toujours disponible.
7. **Overfitting** : tester beaucoup de paramètres finit toujours par trouver un bon résultat par hasard — voir l'onglet Robustesse.
""",
    },
    "en": {
        "title": "📈 Quant Strategy Dashboard — Cross-sectional momentum",
        "expander_explain": "💡 How does this strategy work?",
        "expander_limits": "⚠️ Model limitations",
        "universe": "Asset universe", "search_title": "🔎 Add any asset",
        "search": "Search (name or ticker)", "search_pick": "Added assets",
        "search_ph": "Pick from results or type a ticker",
        "converted": "converted to USD",
        "period": "Period", "date_start": "Start", "date_end": "End", "train_pct": "Train share (%)",
        "signal": "Signal", "lookback": "Momentum lookback (days)", "skip": "Recent days skipped",
        "vol_adj": "Risk-adjusted momentum (÷ volatility)",
        "portfolio": "Portfolio", "n_long": "Assets bought", "allow_short": "Allow short selling",
        "n_short": "Assets shorted", "short_exp": "Short exposure",
        "abs_filter": "Trend filter (long only if momentum > 0, otherwise cash)",
        "weighting": "Weighting", "long_exp": "Invested share (long leg)",
        "use_vt": "Target a volatility", "vt": "Target volatility (annual)", "max_lev": "Max gross leverage",
        "execution": "Execution", "rebal": "Rebalancing", "fee": "Fees (bp per unit traded)",
        "borrow": "Short borrow cost (%/yr)",
        "w_equal": "Equal weight", "w_inverse_vol": "Inverse volatility", "w_risk_parity": "Risk parity",
        "f_D": "Daily", "f_W": "Weekly", "f_M": "Monthly",
        "err_assets": "Select at least 3 assets.", "err_dates": "Start date must be before end date.",
        "err_slots": "Assets bought + shorted must be ≤ number of assets.",
        "loading": "Downloading data…", "computing": "Running backtest…",
        "effective": "Effective period", "limited_by": "start limited by the history of",
        "tab_perf": "Performance", "tab_alloc": "Allocation & cash", "tab_corr": "Correlations & risk",
        "tab_assets": "Per asset", "tab_robust": "Robustness", "tab_data": "Data",
        "strategy": "Strategy", "bench": "Equal weight (100% invested)", "cash_curve": "Cash (risk-free rate)",
        "train": "Train", "test": "Test", "log_scale": "Log scale",
        "equity": "Portfolio value (base 1)", "drawdown": "Drawdown", "roll_sharpe": "6-month rolling Sharpe",
        "headline": "Out-of-sample (test)",
        "m_cagr": "Annualised return", "m_vol": "Volatility", "m_sharpe": "Sharpe", "m_sortino": "Sortino",
        "m_maxdd": "Max drawdown", "m_calmar": "Calmar", "m_win": "Winning days", "m_beta": "Beta vs equal weight",
        "m_alpha": "Annualised alpha", "m_corr": "Correlation vs equal weight", "m_gross": "Avg gross exposure",
        "m_net": "Avg net exposure", "m_cash": "Avg cash", "m_inv": "Time invested",
        "m_turn": "Annual turnover", "m_total": "Total return",
        "weights_title": "Weights held over time (above 0: long, below 0: short)",
        "others": "Others", "cash": "Cash",
        "exposure_title": "Exposure and cash", "gross": "Gross exposure", "net": "Net exposure",
        "current_title": "Target weights at last rebalance", "costs_paid": "Fees paid (total)",
        "corr_title": "Correlation matrix of daily returns", "corr_period": "Period",
        "full": "Full history", "avg_corr_title": "Average pairwise correlation (3-month rolling)",
        "rc_title": "Risk contribution of the current portfolio", "div_ratio": "Diversification ratio",
        "div_help": "Σ|wᵢ|·σᵢ / portfolio σ. 1 = no diversification benefit.",
        "assets_table": "Per-asset statistics (full period)",
        "a_class": "Class", "a_cagr": "Ann. return", "a_vol": "Volatility", "a_sharpe": "Sharpe",
        "a_dd": "Max DD", "a_w": "Avg weight", "a_long": "% days long", "a_short": "% days short",
        "a_contrib": "P&L contribution",
        "robust_intro": "Lookback × number of assets bought grid. The best pair is picked **on train only**, then "
                        "checked on test. A robust parameter sits on a plateau, not an isolated peak.",
        "robust_btn": "Run the grid", "grid_train": "Sharpe — train", "grid_test": "Sharpe — test",
        "best_train": "Best pair (train)", "its_test": "Its test Sharpe",
        "fee_title": "Fee sensitivity (test Sharpe)", "fee_axis": "Fees (bp)",
        "yes": "Yes", "no": "No",
        "pdf_title": "📄 PDF report",
        "pdf_caption": "Includes this page's parameters, metrics and charts (and the walk-forward if it was "
                       "run). Takes ~10 s.",
        "pdf_btn": "Generate PDF report", "pdf_spinner": "Exporting charts and laying out…",
        "pdf_dl": "📥 Download PDF", "pdf_stale": "Parameters changed since generation: regenerate the report.",
        "pdf_error": "Could not generate the PDF",
        "pdf_doc_title": "Backtest report — Cross-sectional momentum", "pdf_assets_n": "assets",
        "pdf_params": "Parameters", "pdf_universe": "Universe", "pdf_results": "Results",
        "pdf_summary": "Over the test period, the strategy shows a Sharpe of {s:.2f} vs {b:.2f} for the equal-"
                       "weight portfolio, with {cagr:.1%} annualised return, {dd:.1%} max drawdown and "
                       "{cash:.0%} average cash.",
        "tab_cmp": "Comparison",
        "cmp_short_cols": ["Ann. ret.", "Vol.", "Sharpe", "Sortino", "Max DD", "Calmar", "Avg cash", "Turnover"],
        "caps_header": "Asset class caps", "caps_on": "Enable caps",
        "caps_help": "Maximum long weight per asset class, as % of the portfolio. The excess is redistributed "
                     "to other selected assets with room left, otherwise it stays in cash.",
        "caps_reassign": "Class of assets added via search",
        "caps_sum_note": "Caps add up to {tot:.0%}: anything above sits in cash.",
        "b_Bourse": "Equities", "b_Obligations": "Bonds", "b_Matières premières": "Commodities",
        "b_Crypto / Autres": "Crypto / Other",
        "class_title": "Long exposure by asset class",
        "s_xs_mom": "Cross-sectional momentum", "s_xs_mom_nocap": "XS momentum without caps",
        "s_ts_mom": "Time-series momentum", "s_strategic": "Strategic allocation {alloc}",
        "s_equal": "Equal weight", "s_risk_parity": "Risk parity",
        "cmp_intro": "Same assets, fees, rebalancing frequency and start date: only the allocation rule changes.\n"
                     "- **Time-series momentum**: each asset is bought (1/N) if *its own* trend is positive, "
                     "otherwise its share stays in cash.\n"
                     "- **Strategic allocation**: the caps become fixed class weights, split equally within each class.\n"
                     "- **Risk parity**: every asset contributes equally to risk, no signal at all.",
        "cmp_intro_pdf": "Same assets, fees, rebalancing and start date; only the allocation rule changes. Momentum "
                         "variants use the caps and circuit breaker; passive benchmarks do not.",
        "cmp_equity": "Strategy value (base 1)", "cmp_scatter": "Return vs volatility",
        "cmp_corr": "Correlation between strategies (daily returns)",
        "cmp_note": "Momentum variants use the sidebar caps and circuit breaker; passive benchmarks (strategic, "
                    "equal weight, risk parity) do not. A class with no selected asset leaves its strategic share in cash.",
        "dd_header": "Circuit breaker", "dd_on": "Enable drawdown circuit breaker",
        "dd_help": "If the portfolio falls more than the threshold below its peak, exposure is cut immediately, "
                   "then restored after the cooldown.",
        "dd_limit": "Drawdown threshold (0.15 = −15%)", "dd_cut": "Exposure kept while cut",
        "dd_cooldown": "Cooldown before re-entry (business days)", "dd_triggers": "Triggers",
        "dd_time": "Of time cut", "dd_shade_note": "red bands on the charts",
        "dd_line": "breaker threshold",
        "tab_wf": "Walk-forward",
        "wf_intro": "At each re-optimisation date, we pick the (lookback, number of assets) pair with the best "
                    "Sharpe over the **past** training window, then apply it until the next date. The stitched "
                    "curve is fully out-of-sample: the most honest estimate of 're-tuning parameters regularly'.",
        "wf_train": "Training window (years)", "wf_step": "Re-optimise every", "months": "months",
        "wf_anchored": "Anchored window", "wf_anchored_help": "Anchored: train from the start of history. "
                       "Rolling: only the last N years.",
        "wf_nl_range": "Assets bought tested (min – max)", "wf_btn": "Run walk-forward",
        "wf_vs_fixed": "vs fixed", "wf_changes": "Parameter changes", "wf_switch": "Switching costs",
        "wf_fixed": "Fixed parameters (sidebar)", "wf_equity": "Walk-forward vs fixed parameters vs equal weight",
        "wf_chosen": "Chosen parameter",
        "wf_promise": "Promised Sharpe (training window) vs realised (next period)",
        "wf_promised": "Promised (train)", "wf_realised": "Realised (next period)",
        "wf_promise_caption": "On average: promised {promised:.2f}, realised {realised:.2f}. A large gap means "
                              "the past winner owed a lot to luck.",
        "dl_returns": "📥 Daily returns (CSV)", "dl_weights": "📥 Weights held (CSV)",
        "dl_prices": "📥 Aligned prices (CSV)",
        "explain_md": """
### The idea: rank assets against each other

**Cross-sectional** momentum doesn't ask "is this asset going up?" but "**which one is going up the most
relative to the others?**". At each rebalance:

| Step | What happens |
|---|---|
| 1 | Each asset's score = performance over the last `lookback` days, skipping the last `skip` days (short-term reversal) |
| 2 | Optional: score ÷ volatility, to compare assets at equal risk |
| 3 | **Buy the top N** and, if allowed, **short the bottom M** |
| 4 | Weight each leg: equal, inverse volatility, or risk parity |
| 5 | Scale exposure: invested share, trend filter, volatility target, max leverage |
| 6 | Whatever is not invested sits in **cash earning the risk-free rate** (US 3-month T-bill) |

Positions decided at day t's close only earn from t+1; fees are charged on what is actually traded.

### Not always 100% invested
- **Invested share** < 100%: the rest is cash.
- **Trend filter**: a top-ranked asset is only bought if its momentum is positive. If everything falls, you end in cash.
- **Volatility target**: exposure shrinks when markets get choppy.
- **Circuit breaker**: if the loss from the peak exceeds a threshold, exposure is cut immediately, then restored after a cooldown.

### Reading the results
- **Sharpe** (over the risk-free rate): > 1 good, 0–1 acceptable, < 0 doesn't beat cash.
- **Equal weight**: naive, always-invested benchmark. A strategy that doesn't beat its Sharpe adds nothing.
- **Train / Test**: tune on train; the unseen test set tells you if it holds.
""",
        "limits_md": """
1. **Execution at the close**: signal and trade use the same closing price — optimistic.
2. **Linear fees**: no variable spread or market impact.
3. **Business-day calendar**: for Bitcoin, Monday's return includes the weekend.
4. **Continuous futures** (`=F`): Yahoo prices ignore roll cost/yield.
5. **Survivorship bias**: the universe is picked today, from assets that survived.
6. **Ideal shorting**: constant borrow cost, always available.
7. **Overfitting**: testing many parameters always finds a good result by luck — see the Robustness tab.
""",
    },
}

with st.sidebar:
    en_mode = st.toggle("🇬🇧 English", value=False, key="k_lang")
T = LANG["en" if en_mode else "fr"]

# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES & CALCULS (mis en cache)
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner=False)
def get_data(tickers: tuple[str, ...], start, end, currencies: tuple = ()):
    prices = load_prices(list(tickers), start, end, currencies=dict(currencies))
    rf = load_risk_free(prices.index, start, end)
    return prices, rf, prices.attrs.get("first_dates", {})


@st.cache_data(show_spinner=False, ttl=3600)
def search_cached(query: str):
    return search_assets(query)


@st.cache_data(show_spinner=False)
def currency_cached(ticker: str) -> str:
    return get_currency(ticker)


@st.cache_data(show_spinner=False)
def backtest(prices, rf, cfg: dict, run_kw: dict):
    return run_backtest(prices, PortfolioConfig(**cfg), rf=rf, **run_kw)


@st.cache_data(show_spinner=False)
def compare_cached(prices, rf, cfg: dict, run_kw: dict, buckets: dict, targets: dict):
    return compare_strategies(prices, PortfolioConfig(**cfg), rf, run_kw, buckets, targets)


@st.cache_data(show_spinner=False)
def walk_forward_cached(prices, rf, cfg: dict, run_kw: dict, grid: list, years, step, anchored):
    return walk_forward(prices, PortfolioConfig(**cfg), grid, rf=rf, run_kw=run_kw,
                        train_years=years, step_months=step, anchored=anchored)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header(T["universe"])
    selected: list[str] = []
    for cat, assets in UNIVERSE.items():
        tickers_cat = list(assets.values())
        with st.expander(cat, expanded=False):
            selected += st.multiselect(
                cat, tickers_cat, default=[t for t in tickers_cat if t in DEFAULT_SELECTION],
                format_func=label, label_visibility="collapsed", key=f"u_{cat}",
            )
    # ── Recherche libre : nom ou ticker -> suggestions Yahoo Finance ──
    with st.expander(T["search_title"], expanded=True):
        query = st.text_input(T["search"], key="k_search", placeholder="Tesla, LVMH, uranium, Bund…").strip()
        found = search_cached(query) if len(query) >= 2 else []
        meta = st.session_state.setdefault("search_meta", {})
        for f in found:
            meta[f["symbol"]] = f
        current = st.session_state.get("k_extra", [])
        options = list(dict.fromkeys(current + [f["symbol"] for f in found]))
        extra = st.multiselect(
            T["search_pick"], options, key="k_extra", accept_new_options=True, placeholder=T["search_ph"],
            format_func=lambda s: (f"{s} — {meta[s]['name']} · {meta[s]['type']} · {meta[s]['exchange']}"
                                   if s in meta else s),
        )
        extra = [s.strip().upper() if s not in meta else s for s in extra]
    for s in extra:
        if s in meta:
            register(s, meta[s]["name"], meta[s]["type"] or "—")
    selected += [s for s in extra if s not in selected]
    currencies = {s: currency_cached(s) for s in extra}

    st.header(T["period"])
    c1, c2 = st.columns(2)
    start_date = c1.date_input(T["date_start"], value=pd.Timestamp("2015-01-01"), key="k_date_start")
    end_date = c2.date_input(T["date_end"], value=pd.Timestamp("today"), key="k_date_end")
    train_pct = st.slider(T["train_pct"], 40, 90, 70, step=5, key="k_train_pct")

    st.header(T["signal"])
    lookback = st.slider(T["lookback"], 21, 252, 126, step=21, key="k_lookback")
    skip = st.slider(T["skip"], 0, 42, 21, step=1, key="k_skip")
    vol_adj = st.checkbox(T["vol_adj"], value=True, key="k_vol_adj")

    st.header(T["portfolio"])
    n_assets = max(len(selected), 1)
    n_long = st.slider(T["n_long"], 1, max(n_assets, 2), min(3, n_assets), key="k_n_long")
    allow_short = st.checkbox(T["allow_short"], value=False, key="k_allow_short")
    n_short, short_exp = 0, 0.0
    if allow_short:
        n_short = st.slider(T["n_short"], 1, max(n_assets, 2), min(2, n_assets), key="k_n_short")
        short_exp = st.slider(T["short_exp"], 0.1, 1.5, 0.5, step=0.1, key="k_short_exp")
    abs_filter = st.checkbox(T["abs_filter"], value=True, key="k_abs_filter")
    weighting = st.selectbox(T["weighting"], ["equal", "inverse_vol", "risk_parity"],
                             format_func=lambda k: T[f"w_{k}"], key="k_weighting")
    long_exp = st.slider(T["long_exp"], 0.1, 1.5, 1.0, step=0.05, key="k_long_exp")
    use_vt = st.checkbox(T["use_vt"], value=False, key="k_use_vt")
    vol_target = st.slider(T["vt"], 0.02, 0.30, 0.10, step=0.01, key="k_vt") if use_vt else None
    max_lev = st.slider(T["max_lev"], 0.5, 3.0, max(1.0, long_exp + short_exp), step=0.1, key="k_max_lev")

    st.header(T["caps_header"])
    caps_on = st.checkbox(T["caps_on"], value=True, key="k_caps_on", help=T["caps_help"])
    buckets = {t: bucket(t) for t in selected}
    extra_only = [t for t in selected if t in extra]
    if extra_only:
        with st.expander(T["caps_reassign"]):
            for t in extra_only:
                buckets[t] = st.selectbox(label(t), BUCKETS, index=BUCKETS.index(buckets[t]), key=f"k_bucket_{t}",
                                          format_func=lambda b: T["b_" + b])
    caps = {}
    for b in BUCKETS:
        n_b = sum(1 for t in selected if buckets[t] == b)
        caps[b] = st.slider(f"{T['b_' + b]} ({n_b} {T['pdf_assets_n']})", 0, 100, int(DEFAULT_CAPS[b] * 100),
                            step=5, format="%d%%", key=f"k_cap_{b}", disabled=not caps_on) / 100
    if caps_on and sum(caps.values()) < long_exp - 1e-9:
        st.caption(T["caps_sum_note"].format(tot=sum(caps.values())))

    st.header(T["execution"])
    rebal = st.selectbox(T["rebal"], ["M", "W", "D"], format_func=lambda k: T[f"f_{k}"], key="k_rebal")
    fee_bp = st.slider(T["fee"], 0.0, 50.0, 5.0, step=0.5, key="k_fee")
    borrow = st.slider(T["borrow"], 0.0, 5.0, 1.0, step=0.25, key="k_borrow") / 100 if allow_short else 0.0

    st.header(T["dd_header"])
    dd_on = st.checkbox(T["dd_on"], value=False, key="k_dd_on", help=T["dd_help"])
    dd_limit, dd_cut, dd_cooldown = None, 0.0, 21
    if dd_on:
        dd_limit = st.slider(T["dd_limit"], 0.05, 0.50, 0.15, step=0.01, format="%.2f", key="k_dd_limit")
        dd_cut = st.slider(T["dd_cut"], 0.0, 1.0, 0.0, step=0.05, key="k_dd_cut")
        dd_cooldown = st.slider(T["dd_cooldown"], 5, 126, 21, step=1, key="k_dd_cooldown")

# ══════════════════════════════════════════════════════════════════════════════
# EN-TÊTE + VALIDATIONS
# ══════════════════════════════════════════════════════════════════════════════

st.title(T["title"])
with st.expander(T["expander_explain"]):
    st.markdown(T["explain_md"])
with st.expander(T["expander_limits"]):
    st.markdown(T["limits_md"])

if len(selected) < 3:
    st.error(T["err_assets"]); st.stop()
if start_date >= end_date:
    st.error(T["err_dates"]); st.stop()
if n_long + n_short > len(selected):
    st.error(T["err_slots"]); st.stop()

with st.spinner(T["loading"]):
    try:
        prices, rf, first_dates = get_data(tuple(selected), start_date, end_date,
                                           tuple(sorted(currencies.items())))
    except ValueError as e:
        st.error(str(e)); st.stop()

cfg = dict(n_long=n_long, n_short=n_short, absolute_filter=abs_filter, weighting=weighting,
           long_exposure=long_exp, short_exposure=short_exp, vol_target=vol_target, max_leverage=max_lev,
           class_caps=caps if caps_on else None, asset_class=buckets)
fee = fee_bp / 10_000
run_kw = dict(lookback=lookback, skip=skip, vol_adjusted=vol_adj, rebalance=rebal, fee=fee, borrow_cost=borrow,
              dd_limit=dd_limit, dd_cut=dd_cut, dd_cooldown=dd_cooldown)
with st.spinner(T["computing"]):
    res = backtest(prices, rf, cfg, run_kw)
bench = equal_weight_benchmark(prices)


def breaker_periods(flag: pd.Series) -> list[tuple]:
    """Intervalles [début, fin] où le coupe-circuit est actif."""
    if flag is None or not flag.any():
        return []
    f = flag.astype(int)
    starts = flag.index[(f.diff() == 1) | ((f == 1) & (np.arange(len(f)) == 0))]
    ends = flag.index[(f.diff(-1) == 1) | ((f == 1) & (np.arange(len(f)) == len(f) - 1))]
    return list(zip(starts, ends))


def shade_breaker(fig: go.Figure):
    for a, b in breaker_periods(res.breaker):
        fig.add_vrect(x0=a, x1=b, fillcolor="#d03b3b", opacity=0.12, line_width=0)


BREAKS = breaker_periods(res.breaker)

# Train / test : un seul backtest sur tout l'historique, puis découpage (pas de « démarrage à froid » du test)
first_live = res.rebalance_dates[0] if len(res.rebalance_dates) else prices.index[0]
live_idx = prices.index[prices.index >= first_live]
split_date = live_idx[int(len(live_idx) * train_pct / 100)]
train = slice(first_live, split_date - pd.Timedelta(days=1))
test = slice(split_date, None)

limiting = max(first_dates, key=lambda t: first_dates[t]) if first_dates else None
caption = (f"{T['effective']} : {prices.index[0].date()} → {prices.index[-1].date()} · "
           f"{len(selected)} actifs · {T['train']} → {(split_date - pd.Timedelta(days=1)).date()} · "
           f"{T['test']} {split_date.date()} →")
if limiting and pd.Timestamp(first_dates[limiting]) > pd.Timestamp(start_date) + pd.Timedelta(days=10):
    caption += f"  ·  ⚠️ {T['limited_by']} **{label(limiting)}**"
conv = [f"{label(t)} ({c})" for t, c in currencies.items() if c != "USD"]
if conv:
    caption += f"  ·  💱 {', '.join(conv)} {T['converted']}"
st.caption(caption)


def metrics(r: pd.Series, s: slice, with_book=True) -> dict:
    return performance_metrics(
        r.loc[s], rf, benchmark=bench.loc[s],
        weights=res.weights.loc[s] if with_book else None,
        turnover=res.turnover.loc[s] if with_book else None,
    )


m_tr, m_te = metrics(res.returns, train), metrics(res.returns, test)
b_tr, b_te = metrics(bench, train, False), metrics(bench, test, False)

PCT = {"total_return", "cagr", "ann_vol", "max_dd", "win_rate", "alpha_ann", "avg_cash", "time_invested",
       "avg_gross_exposure", "avg_net_exposure", "ann_turnover"}
ROWS = [("total_return", "m_total"), ("cagr", "m_cagr"), ("ann_vol", "m_vol"), ("sharpe", "m_sharpe"),
        ("sortino", "m_sortino"), ("max_dd", "m_maxdd"), ("calmar", "m_calmar"), ("win_rate", "m_win"),
        ("beta", "m_beta"), ("alpha_ann", "m_alpha"), ("correlation_bench", "m_corr"),
        ("avg_gross_exposure", "m_gross"), ("avg_net_exposure", "m_net"), ("avg_cash", "m_cash"),
        ("time_invested", "m_inv"), ("ann_turnover", "m_turn")]


def fmt(key, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.1%}" if key in PCT else f"{v:.2f}"


def _plain(md: str) -> str:
    """Markdown simple -> texte brut pour le PDF."""
    out = []
    for line in md.strip().splitlines():
        line = line.replace("**", "").replace("`", "").lstrip("#").strip()
        if line.startswith("|") or set(line) <= {"-", " ", "|"}:
            continue
        out.append(line)
    return "\n".join(out)


def make_report() -> bytes:
    figs = REPORT["figs"]
    sub = (f"{T['effective']} : {prices.index[0].date()} → {prices.index[-1].date()} · {len(selected)} "
           f"{T['pdf_assets_n']} · {T['train']} → {(split_date - pd.Timedelta(days=1)).date()} · "
           f"{T['test']} {split_date.date()} →")
    params = {
        T["lookback"]: lookback, T["skip"]: skip, T["vol_adj"]: T["yes"] if vol_adj else T["no"],
        T["n_long"]: n_long, T["n_short"]: n_short if allow_short else "—",
        T["short_exp"]: f"{short_exp:.0%}" if allow_short else "—",
        T["abs_filter"]: T["yes"] if abs_filter else T["no"], T["weighting"]: T[f"w_{weighting}"],
        T["long_exp"]: f"{long_exp:.0%}", T["vt"]: f"{vol_target:.0%}" if vol_target else "—",
        T["max_lev"]: f"{max_lev:.1f}", T["rebal"]: T[f"f_{rebal}"], T["fee"]: f"{fee_bp:.1f}",
        T["borrow"]: f"{borrow:.2%}" if allow_short else "—",
        T["dd_header"]: (f"{dd_limit:.0%} · {dd_cut:.0%} · {dd_cooldown} j" if dd_limit else "—"),
        T["caps_header"]: (" / ".join(f"{T['b_' + b]} {caps[b]:.0%}" for b in BUCKETS) if caps_on else "—"),
    }
    universe = ", ".join(f"{label(t)} [{TICKER_TO_CLASS.get(t, '—')}]" for t in selected)
    m_s, m_b = m_te.get("sharpe", np.nan), b_te.get("sharpe", np.nan)
    secs = [
        ("h1", T["pdf_params"]), ("kv", params),
        ("h1", T["pdf_universe"]), ("text", universe),
        ("h1", T["pdf_results"]),
        ("text", T["pdf_summary"].format(s=m_s, b=m_b, cagr=m_te.get("cagr", np.nan),
                                         dd=m_te.get("max_dd", np.nan), cash=m_te.get("avg_cash", np.nan))),
        ("table", REPORT["metrics"]),
        ("pagebreak", None),
        ("h1", T["tab_cmp"]), ("text", T["cmp_intro_pdf"]), ("table", REPORT["cmp"], 0.31),
        ("fig", figs["cmp_equity"], 440), ("fig", figs["cmp_scatter"], 380),
        ("pagebreak", None),
        ("fig", figs["equity"], 430), ("fig", figs["drawdown"], 340), ("fig", figs["roll_sharpe"], 340),
        ("pagebreak", None),
        ("h1", T["tab_alloc"]), ("fig", figs["weights"], 430), ("fig", figs["exposure"], 340),
        ("fig", figs["classes"], 340),
    ]
    if dd_limit:
        n_cut = int((res.breaker_events["event"] == "coupure").sum())
        secs.append(("text", f"{T['dd_header']} : {n_cut} {T['dd_triggers'].lower()} · "
                             f"{res.breaker.loc[first_live:].mean():.1%} {T['dd_time'].lower()}."))
    if "current" in figs:
        secs.append(("fig", figs["current"], 380))
    secs += [("pagebreak", None), ("h1", T["tab_corr"]), ("fig", figs["corr"], max(420, 32 * len(selected) + 120))]
    secs += [("fig", figs[k], 340) for k in ("avg_corr", "rc") if k in figs]
    a = REPORT["assets"].copy()
    a.index = [str(i)[:30] for i in a.index]
    a = a.drop(columns=[T["a_class"]] + ([] if allow_short else [T["a_short"]]))
    for c in a.columns:
        a[c] = a[c].map("{:.2f}".format if c == T["a_sharpe"] else "{:.1%}".format)
    secs += [("pagebreak", None), ("h1", T["assets_table"]), ("table", a)]
    if "wf" in REPORT and "wf_equity" in figs:
        wfr = REPORT["wf"]
        secs += [("pagebreak", None), ("h1", T["tab_wf"]), ("text", wfr["metrics"]),
                 ("fig", figs["wf_equity"], 400), ("fig", figs["wf_promise"], 340), ("text", wfr["summary"]),
                 ("table", wfr["table"].astype(str))]
    secs += [("h1", T["expander_limits"].split(" ", 1)[1]), ("text", _plain(T["limits_md"]))]
    return build_pdf(T["pdf_doc_title"], sub, secs)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([T["tab_perf"], T["tab_cmp"], T["tab_alloc"], T["tab_corr"], T["tab_assets"], T["tab_robust"], T["tab_wf"], T["tab_data"]])

# ── 1. PERFORMANCE ────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader(T["headline"])
    k = st.columns(5)
    for col, key, lbl in zip(k, ["sharpe", "cagr", "ann_vol", "max_dd", "avg_cash"],
                             ["m_sharpe", "m_cagr", "m_vol", "m_maxdd", "m_cash"]):
        delta = None
        if key in b_te:
            d = m_te.get(key, np.nan) - b_te[key]
            delta = f"{d:+.1%} vs EW" if key in PCT else f"{d:+.2f} vs EW"
        col.metric(T[lbl], fmt(key, m_te.get(key)), delta=delta,
                   delta_color="inverse" if key == "ann_vol" else "normal")

    s_live = slice(first_live, None)
    eq_s = (1 + res.returns.loc[s_live]).cumprod()
    eq_b = (1 + bench.loc[s_live]).cumprod()
    eq_c = (1 + rf.loc[s_live]).cumprod()
    if dd_limit:
        n_cut = int((res.breaker_events["event"] == "coupure").sum())
        st.info(f"🛑 {T['dd_header']} : {n_cut} {T['dd_triggers'].lower()} · "
                f"{res.breaker.loc[first_live:].mean():.0%} {T['dd_time'].lower()} — {T['dd_shade_note']}")
    log_y = st.toggle(T["log_scale"], value=False, key="k_log_scale")
    fig = go.Figure()
    fig.add_vrect(x0=split_date, x1=eq_s.index[-1], fillcolor=NEUTRAL, opacity=0.08, line_width=0,
                  annotation_text=T["test"], annotation_position="top left")
    fig.add_trace(go.Scatter(x=eq_s.index, y=eq_s, name=T["strategy"], line=dict(color=C_STRAT, width=2)))
    fig.add_trace(go.Scatter(x=eq_b.index, y=eq_b, name=T["bench"], line=dict(color=C_BENCH, width=2)))
    fig.add_trace(go.Scatter(x=eq_c.index, y=eq_c, name=T["cash_curve"],
                             line=dict(color=NEUTRAL, width=1.5, dash="dot")))
    shade_breaker(fig)
    style(fig, T["equity"], y_fmt=".2f", height=430)
    if log_y:
        fig.update_yaxes(type="log")
    show(fig, "equity")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for r, name, color in [(res.returns, T["strategy"], C_STRAT), (bench, T["bench"], C_BENCH)]:
            dd = drawdown_series(r.loc[s_live])
            fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, line=dict(color=color, width=1.5)))
        if dd_limit:
            fig.add_hline(y=-dd_limit, line_dash="dash", line_color="#d03b3b", line_width=1,
                          annotation_text=T["dd_line"], annotation_position="bottom left")
        shade_breaker(fig)
        show(style(fig, T["drawdown"], y_fmt=".0%"), "drawdown")
    with c2:
        fig = go.Figure()
        for r, name, color in [(res.returns, T["strategy"], C_STRAT), (bench, T["bench"], C_BENCH)]:
            ex = (r - rf).loc[s_live]
            rs = ex.rolling(126).mean() / ex.rolling(126).std() * np.sqrt(TRADING_DAYS)
            fig.add_trace(go.Scatter(x=rs.index, y=rs, name=name, line=dict(color=color, width=1.5)))
        show(style(fig, T["roll_sharpe"], y_fmt=".1f"), "roll_sharpe")

    table = pd.DataFrame(
        {f"{T['strategy']} — {T['train']}": [fmt(k_, m_tr.get(k_)) for k_, _ in ROWS],
         f"{T['strategy']} — {T['test']}": [fmt(k_, m_te.get(k_)) for k_, _ in ROWS],
         f"EW — {T['train']}": [fmt(k_, b_tr.get(k_)) for k_, _ in ROWS],
         f"EW — {T['test']}": [fmt(k_, b_te.get(k_)) for k_, _ in ROWS]},
        index=[T[l] for _, l in ROWS],
    )
    st.dataframe(table, width="stretch")
    REPORT["metrics"] = table

# ── 2. COMPARAISON DE STRATÉGIES ─────────────────────────────────────────────
with tabs[1]:
    st.markdown(T["cmp_intro"])
    with st.spinner(T["computing"]):
        strats = compare_cached(prices, rf, cfg, run_kw, buckets, caps)
    strats = {k: v for k, v in strats.items() if k in STRATEGY_KEYS}
    s_names = {k: T[f"s_{k}"].format(alloc="/".join(f"{caps[b] * 100:.0f}" for b in BUCKETS)) for k in strats}
    s_color = {k: SERIES[i] for i, k in enumerate(STRATEGY_KEYS)}
    cmp_per = st.radio(T["corr_period"], ["test", "full"], format_func=lambda k: T[k], horizontal=True,
                       key="k_cmp_period")
    cs = test if cmp_per == "test" else slice(first_live, None)

    rows = {}
    for k, r in strats.items():
        m = performance_metrics(r.returns.loc[cs], rf, weights=r.weights.loc[cs], turnover=r.turnover.loc[cs])
        rows[s_names[k]] = {T["m_cagr"]: m["cagr"], T["m_vol"]: m["ann_vol"], T["m_sharpe"]: m["sharpe"],
                            T["m_sortino"]: m["sortino"], T["m_maxdd"]: m["max_dd"], T["m_calmar"]: m["calmar"],
                            T["m_cash"]: m["avg_cash"], T["m_turn"]: m["ann_turnover"]}
    cmp_df = pd.DataFrame(rows).T
    ratio_cols = [T["m_sharpe"], T["m_sortino"], T["m_calmar"]]
    st.dataframe(cmp_df.style.format({c: ("{:.2f}" if c in ratio_cols else "{:.1%}") for c in cmp_df.columns})
                 .highlight_max(subset=[T["m_sharpe"]], props="font-weight: 700"), width="stretch")
    REPORT["cmp"] = cmp_df.apply(lambda col: col.map(("{:.2f}" if col.name in ratio_cols else "{:.1%}").format))         .set_axis(T["cmp_short_cols"], axis=1)   # intitulés courts : le tableau tient sur une page A4

    fig = go.Figure()
    for k, r in strats.items():
        eq = (1 + r.returns.loc[cs]).cumprod()
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=s_names[k],
                                 line=dict(color=s_color[k], width=2.5 if k == "xs_mom" else 1.5)))
    show(style(fig, T["cmp_equity"], y_fmt=".2f", height=440), "cmp_equity")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for k, r in strats.items():
            dd = drawdown_series(r.returns.loc[cs])
            fig.add_trace(go.Scatter(x=dd.index, y=dd, name=s_names[k], line=dict(color=s_color[k], width=1.2)))
        fig.update_layout(showlegend=False)
        show(style(fig, T["drawdown"], y_fmt=".0%"), "cmp_dd")
    with c2:
        fig = go.Figure()
        placed = []   # étiquettes au-dessus, ou en dessous si un point voisin occupe déjà la place
        for k in strats:
            row = cmp_df.loc[s_names[k]]
            x_, y_ = row[T["m_vol"]], row[T["m_cagr"]]
            pos = ("bottom center" if any(abs(x_ - a) < 0.015 and abs(y_ - b) < 0.012 and q == "top center"
                                          for a, b, q in placed) else "top center")
            placed.append((x_, y_, pos))
            fig.add_trace(go.Scatter(
                x=[x_], y=[y_], mode="markers+text", name=s_names[k],
                text=[s_names[k]], textposition=pos, textfont=dict(size=10),
                marker=dict(size=13, color=s_color[k], line=dict(width=2, color="white")),
                hovertemplate=f"{s_names[k]}<br>{T['m_vol']} %{{x:.1%}}<br>{T['m_cagr']} %{{y:.1%}}<extra></extra>"))
        style(fig, T["cmp_scatter"])
        fig.update_layout(showlegend=False, hovermode="closest")
        xv, yv = cmp_df[T["m_vol"]], cmp_df[T["m_cagr"]]
        pad_x, pad_y = (xv.max() - xv.min()) * 0.25 + 0.005, (yv.max() - yv.min()) * 0.15 + 0.005
        fig.update_xaxes(title=T["m_vol"], tickformat=".0%", range=[xv.min() - pad_x, xv.max() + pad_x])
        fig.update_yaxes(title=T["m_cagr"], tickformat=".0%", range=[yv.min() - pad_y, yv.max() + pad_y])
        show(fig, "cmp_scatter")

    corr_s = pd.DataFrame({s_names[k]: r.returns.loc[cs] for k, r in strats.items()}).corr()
    fig = go.Figure(go.Heatmap(
        z=corr_s.values, x=list(corr_s.columns), y=list(corr_s.index), zmin=-1, zmax=1, zmid=0,
        colorscale=DIVERGING, text=np.round(corr_s.values, 2), texttemplate="%{text}", xgap=2, ygap=2,
        hovertemplate="%{y} / %{x} : %{z:.2f}<extra></extra>"))
    style(fig, T["cmp_corr"], height=420)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False)
    show(fig, "cmp_corr")
    st.caption(T["cmp_note"])

# ── 3. ALLOCATION & CASH ──────────────────────────────────────────────────────
with tabs[2]:
    W = res.weights.loc[s_live]
    # 7 actifs les plus utilisés + "Autres" (plafond de lisibilité de la palette)
    top = W.abs().mean().sort_values(ascending=False)
    top = list(top[top > 0].index[:7])
    rest = [c for c in W.columns if c not in top]
    groups = {label(t): W[t] for t in top}
    if rest:
        groups[T["others"]] = W[rest].sum(axis=1) if len(rest) > 1 else W[rest[0]]
    Wd = W.resample("W").last()  # hebdomadaire : suffisant à l'œil, bien plus léger
    fig = go.Figure()
    for i, (name, _) in enumerate(groups.items()):
        cols = [t for t in top if label(t) == name] or rest
        series = Wd[cols].sum(axis=1)
        color = SERIES[i] if name != T["others"] else NEUTRAL
        for part, sg in [(series.clip(lower=0), "long"), (series.clip(upper=0), "short")]:
            if (part != 0).any():
                fig.add_trace(go.Scatter(
                    x=part.index, y=part, name=name, legendgroup=name, showlegend=bool(sg == "long" or
                    (series.clip(lower=0) == 0).all()), stackgroup=sg, mode="lines",
                    line=dict(width=0.5, color=color), fillcolor=color,
                    hovertemplate="%{y:.1%}"))
    show(style(fig, T["weights_title"], y_fmt=".0%", height=430), "weights")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=W.index, y=W.abs().sum(axis=1), name=T["gross"],
                                 line=dict(color=C_STRAT, width=1.5)))
        fig.add_trace(go.Scatter(x=W.index, y=W.sum(axis=1), name=T["net"], line=dict(color=C_BENCH, width=1.5)))
        fig.add_trace(go.Scatter(x=W.index, y=res.cash.loc[s_live], name=T["cash"],
                                 line=dict(color=NEUTRAL, width=1.5, dash="dot")))
        shade_breaker(fig)
        show(style(fig, T["exposure_title"], y_fmt=".0%"), "exposure")
    by_class = W.clip(lower=0).T.groupby(pd.Series(buckets)).sum().T.resample("W").last()
    fig = go.Figure()
    for i, b in enumerate(BUCKETS):
        if b in by_class and (by_class[b] > 0).any():
            name_b = T["b_" + b] + (f" (max {caps[b]:.0%})" if caps_on else "")
            fig.add_trace(go.Scatter(x=by_class.index, y=by_class[b], name=name_b, stackgroup="cls",
                                     mode="lines", line=dict(width=0.5, color=SERIES[i]), fillcolor=SERIES[i],
                                     hovertemplate="%{y:.1%}"))
    show(style(fig, T["class_title"], y_fmt=".0%", height=360), "classes")
    with c2:
        last_reb = res.rebalance_dates[-1]
        pos = res.weights.index.get_loc(last_reb)
        tgt = res.weights.iloc[pos + 1] if pos + 1 < len(res.weights) else res.weights.iloc[pos]
        tgt = tgt[tgt.abs() > 1e-6].sort_values()
        cash_w = 1 - tgt.sum()
        bars = pd.concat([tgt.rename(index=label), pd.Series({T["cash"]: cash_w})])
        fig = go.Figure(go.Bar(
            x=bars.values, y=bars.index, orientation="h",
            marker_color=[NEUTRAL if i == T["cash"] else (C_STRAT if v >= 0 else SERIES[7])
                          for i, v in bars.items()],
            text=[f"{v:.1%}" for v in bars.values], textposition="outside", hovertemplate="%{x:.1%}<extra></extra>"))
        style(fig, f"{T['current_title']} ({last_reb.date()})", height=380)
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(hovermode="closest")
        show(fig, "current")
        st.metric(T["costs_paid"], f"{res.costs.sum():.2%}")
        if dd_limit:
            m1, m2 = st.columns(2)
            m1.metric(T["dd_triggers"], int((res.breaker_events["event"] == "coupure").sum()))
            m2.metric(T["dd_time"], f"{res.breaker.loc[first_live:].mean():.1%}")

# ── 3. CORRÉLATIONS & RISQUE ──────────────────────────────────────────────────
with tabs[3]:
    rets = prices.pct_change().dropna()
    per = st.radio(T["corr_period"], ["full", "train", "test"], format_func=lambda k: T[k],
                   horizontal=True, key="k_corr_period")
    sub = {"full": rets, "train": rets.loc[train], "test": rets.loc[test]}[per]
    corr = sub.corr()
    names = [label(t) for t in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=names, y=names, zmin=-1, zmax=1, zmid=0, colorscale=DIVERGING,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y} / %{x} : %{z:.2f}<extra></extra>", xgap=2, ygap=2))
    style(fig, T["corr_title"], height=max(420, 32 * len(names) + 120))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False)
    show(fig, "corr")

    c1, c2 = st.columns([3, 2])
    with c1:
        ac = average_rolling_correlation(rets, window=63)
        fig = go.Figure(go.Scatter(x=ac.index, y=ac, name=T["avg_corr_title"], line=dict(color=C_STRAT, width=1.5)))
        show(style(fig, T["avg_corr_title"], y_fmt=".2f"), "avg_corr")
    with c2:
        w_now = res.weights.iloc[-1]
        hist = rets.iloc[-63:]
        rc = risk_contributions(w_now, hist).sort_values()
        if rc.empty:
            st.info(T["cash"] + " 100 %")
        else:
            fig = go.Figure(go.Bar(x=rc.values, y=[label(t) for t in rc.index], orientation="h",
                                   marker_color=C_STRAT, text=[f"{v:.0%}" for v in rc.values],
                                   textposition="outside", hovertemplate="%{x:.1%}<extra></extra>"))
            style(fig, T["rc_title"])
            fig.update_xaxes(tickformat=".0%")
            fig.update_layout(hovermode="closest")
            show(fig, "rc")
            st.metric(T["div_ratio"], f"{diversification_ratio(w_now, hist):.2f}", help=T["div_help"])

# ── 4. PAR ACTIF ──────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader(T["assets_table"])
    r_live = res.asset_returns.loc[s_live]
    rows = []
    for t in prices.columns:
        m = performance_metrics(r_live[t], rf)
        w = W[t]
        rows.append({
            "Asset" if en_mode else "Actif": label(t), T["a_class"]: TICKER_TO_CLASS.get(t, "—"),
            T["a_cagr"]: m["cagr"], T["a_vol"]: m["ann_vol"], T["a_sharpe"]: m["sharpe"],
            T["a_dd"]: m["max_dd"], T["a_w"]: w.mean(), T["a_long"]: (w > 1e-9).mean(),
            T["a_short"]: (w < -1e-9).mean(), T["a_contrib"]: (w * r_live[t]).sum(),
        })
    df_assets = pd.DataFrame(rows).set_index("Asset" if en_mode else "Actif").sort_values(T["a_contrib"], ascending=False)
    pct_cols = [T["a_cagr"], T["a_vol"], T["a_dd"], T["a_w"], T["a_long"], T["a_short"], T["a_contrib"]]
    REPORT["assets"] = df_assets
    st.dataframe(df_assets.style.format({**{c: "{:.1%}" for c in pct_cols}, T["a_sharpe"]: "{:.2f}"}),
                 width="stretch", height=min(40 * len(df_assets) + 40, 700))

# ── 5. ROBUSTESSE ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown(T["robust_intro"])
    if st.button(T["robust_btn"]):
        lookbacks = [21, 42, 63, 126, 189, 252]
        longs = list(range(1, min(len(selected) - n_short, 6) + 1))
        g_tr = pd.DataFrame(index=lookbacks, columns=longs, dtype=float)
        g_te = g_tr.copy()
        bar = st.progress(0.0)
        total = len(lookbacks) * len(longs)
        for i, lb in enumerate(lookbacks):
            for j, nl in enumerate(longs):
                r = run_backtest(prices, PortfolioConfig(**{**cfg, "n_long": nl}), rf=rf,
                                 **{**run_kw, "lookback": lb, "skip": min(skip, lb - 1)})
                g_tr.loc[lb, nl] = sharpe(r.returns.loc[train], rf)
                g_te.loc[lb, nl] = sharpe(r.returns.loc[test], rf)
                bar.progress((i * len(longs) + j + 1) / total)
        bar.empty()
        best_lb, best_nl = g_tr.stack().idxmax()
        lim = float(np.nanmax(np.abs(np.r_[g_tr.values.ravel(), g_te.values.ravel()])))

        c1, c2 = st.columns(2)
        for col, g, title in [(c1, g_tr, T["grid_train"]), (c2, g_te, T["grid_test"])]:
            with col:
                fig = go.Figure(go.Heatmap(
                    z=g.values, x=[str(c) for c in g.columns], y=[str(i) for i in g.index],
                    zmin=-lim, zmax=lim, zmid=0, colorscale=DIVERGING, xgap=2, ygap=2,
                    text=np.round(g.values.astype(float), 2), texttemplate="%{text}",
                    hovertemplate=f"{T['lookback']} %{{y}} · {T['n_long']} %{{x}} : %{{z:.2f}}<extra></extra>"))
                style(fig, title)
                fig.update_xaxes(title=T["n_long"])
                fig.update_yaxes(title=T["lookback"], showgrid=False, zeroline=False)
                fig.update_layout(hovermode="closest")
                show(fig)
        b1, b2 = st.columns(2)
        b1.metric(T["best_train"], f"lookback {best_lb} · N = {best_nl}", f"Sharpe {g_tr.loc[best_lb, best_nl]:.2f}")
        b2.metric(T["its_test"], f"{g_te.loc[best_lb, best_nl]:.2f}",
                  f"{g_te.loc[best_lb, best_nl] - g_tr.loc[best_lb, best_nl]:+.2f}")

        fees_bp = [0, 2, 5, 10, 20, 30, 50]
        s_fee = [sharpe(run_backtest(prices, PortfolioConfig(**cfg), rf=rf, **{**run_kw, "fee": f / 10_000})
                        .returns.loc[test], rf) for f in fees_bp]
        fig = go.Figure(go.Scatter(x=fees_bp, y=s_fee, mode="lines+markers", name=T["m_sharpe"],
                                   line=dict(color=C_STRAT, width=2), marker=dict(size=8)))
        style(fig, T["fee_title"], y_fmt=".2f")
        fig.update_xaxes(title=T["fee_axis"])
        fig.update_layout(hovermode="closest")
        show(fig)

# ── 6. WALK-FORWARD ───────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown(T["wf_intro"])
    c1, c2, c3 = st.columns(3)
    wf_years = c1.slider(T["wf_train"], 1.0, 5.0, 3.0, step=0.5, key="k_wf_years")
    wf_step = c2.selectbox(T["wf_step"], [1, 3, 6, 12], index=2, key="k_wf_step",
                           format_func=lambda m: f"{m} {T['months']}")
    wf_anch = c3.checkbox(T["wf_anchored"], value=False, key="k_wf_anch", help=T["wf_anchored_help"])
    c1, c2 = st.columns(2)
    wf_lbs = c1.multiselect(T["lookback"], [21, 42, 63, 126, 189, 252], default=[21, 42, 63, 126, 189, 252],
                            key="k_wf_lbs")
    max_nl = max(2, len(selected) - n_short)
    wf_nl = c2.slider(T["wf_nl_range"], 1, max_nl, (1, min(6, max_nl)), key="k_wf_nl")
    if st.button(T["wf_btn"], key="k_wf_btn"):
        st.session_state["wf_on"] = True

    if st.session_state.get("wf_on") and wf_lbs:
        grid = param_grid(lookback=sorted(wf_lbs), n_long=list(range(wf_nl[0], wf_nl[1] + 1)))
        try:
            with st.spinner(T["computing"] + f" ({len(grid)} combinaisons)"):
                wf = walk_forward_cached(prices, rf, cfg, run_kw, grid, wf_years, wf_step, wf_anch)
        except ValueError as e:
            st.error(str(e))
        else:
            per = wf.returns.index
            fixed_r = res.returns.loc[per]
            bench_r = bench.loc[per]
            m_wf = performance_metrics(wf.returns, rf)
            m_fx = performance_metrics(fixed_r, rf)
            m_bw = performance_metrics(bench_r, rf)
            k = st.columns(5)
            k[0].metric(T["m_sharpe"] + " WF", f"{m_wf['sharpe']:.2f}",
                        f"{m_wf['sharpe'] - m_fx['sharpe']:+.2f} {T['wf_vs_fixed']}")
            k[1].metric(T["m_cagr"] + " WF", f"{m_wf['cagr']:.1%}", f"{m_wf['cagr'] - m_bw['cagr']:+.1%} vs EW")
            k[2].metric(T["m_maxdd"] + " WF", f"{m_wf['max_dd']:.1%}")
            n_changes = int((wf.choices[["lookback", "n_long"]].diff().abs().sum(axis=1) > 0).sum())
            k[3].metric(T["wf_changes"], f"{n_changes} / {len(wf.choices) - 1}")
            k[4].metric(T["wf_switch"], f"{wf.switch_costs.sum():.2%}")

            fig = go.Figure()
            for d in wf.choices.index:
                fig.add_vline(x=d, line_width=0.5, line_dash="dot", line_color=NEUTRAL)
            for r_, name, color in [(wf.returns, "Walk-forward", C_STRAT), (fixed_r, T["wf_fixed"], SERIES[2]),
                                    (bench_r, T["bench"], C_BENCH)]:
                eq = (1 + r_).cumprod()
                fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name, line=dict(color=color, width=2)))
            show(style(fig, T["wf_equity"], y_fmt=".2f", height=420), "wf_equity")

            c1, c2 = st.columns(2)
            for col, key_, title in [(c1, "lookback", T["lookback"]), (c2, "n_long", T["n_long"])]:
                with col:
                    ch = wf.choices[key_]
                    fig = go.Figure(go.Scatter(
                        x=list(ch.index) + [per[-1]], y=list(ch.values) + [ch.values[-1]], mode="lines+markers",
                        line=dict(color=C_STRAT, width=2, shape="hv"), marker=dict(size=8), name=title))
                    show(style(fig, f"{T['wf_chosen']} : {title}", height=300))

            # Sharpe « promis » sur la fenêtre d'entraînement vs obtenu sur la période suivante
            bounds = list(wf.choices.index) + [per[-1] + pd.Timedelta(days=1)]
            realised = [sharpe(wf.returns.loc[bounds[i]: bounds[i + 1] - pd.Timedelta(days=1)], rf)
                        for i in range(len(wf.choices))]
            xs = [d.strftime("%Y-%m") for d in wf.choices.index]
            fig = go.Figure([
                go.Bar(x=xs, y=wf.choices["train_sharpe"], name=T["wf_promised"], marker_color=SERIES[0]),
                go.Bar(x=xs, y=realised, name=T["wf_realised"], marker_color=SERIES[1]),
            ])
            style(fig, T["wf_promise"], y_fmt=".1f", height=360)
            fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
            show(fig, "wf_promise")
            st.caption(T["wf_promise_caption"].format(
                promised=float(wf.choices["train_sharpe"].mean()), realised=float(np.nanmean(realised))))

            tbl = wf.choices.copy()
            tbl[T["wf_realised"]] = realised
            tbl.index = tbl.index.date
            st.dataframe(tbl.style.format({"train_sharpe": "{:.2f}", T["wf_realised"]: "{:.2f}"}), width="stretch")
            REPORT["wf"] = dict(
                table=tbl.assign(train_sharpe=tbl["train_sharpe"].map("{:.2f}".format),
                                 **{T["wf_realised"]: tbl[T["wf_realised"]].map("{:.2f}".format)})
                .rename(columns={"lookback": T["lookback"], "n_long": T["n_long"],
                                 "train_sharpe": T["wf_promised"]}),
                summary=T["wf_promise_caption"].format(promised=float(wf.choices["train_sharpe"].mean()),
                                                       realised=float(np.nanmean(realised))),
                metrics=f"Sharpe WF {m_wf['sharpe']:.2f} · {T['wf_fixed']} {m_fx['sharpe']:.2f} · "
                        f"EW {m_bw['sharpe']:.2f} · {T['wf_changes']} {n_changes}/{len(wf.choices) - 1} · "
                        f"{T['wf_train']} {wf_years} · {T['wf_step']} {wf_step} {T['months']}",
            )

# ── 7. DONNÉES ────────────────────────────────────────────────────────────────
with tabs[7]:
    out = pd.DataFrame({"strategy": res.returns, "equal_weight": bench, "rf": rf,
                        "costs": res.costs, "cash": res.cash, "breaker": res.breaker.astype(float)})
    c1, c2, c3 = st.columns(3)
    c1.download_button(T["dl_returns"], out.to_csv().encode(), "returns.csv", "text/csv")
    c2.download_button(T["dl_weights"], res.weights.to_csv().encode(), "weights.csv", "text/csv")
    c3.download_button(T["dl_prices"], prices.to_csv().encode(), "prices.csv", "text/csv")
    st.dataframe(out.tail(30).style.format("{:.4%}"), width="stretch")

    st.subheader(T["pdf_title"])
    st.caption(T["pdf_caption"])
    signature = repr((selected, str(start_date), str(end_date), train_pct, cfg, run_kw, en_mode,
                      "wf" in REPORT))
    if st.button(T["pdf_btn"], key="k_pdf_btn", type="primary"):
        with st.spinner(T["pdf_spinner"]):
            try:
                st.session_state["pdf"] = (signature, make_report())
            except Exception as e:  # ex. Chrome absent pour l'export des graphiques
                st.error(f"{T['pdf_error']} : {e}")
    if st.session_state.get("pdf"):
        sig, pdf_bytes = st.session_state["pdf"]
        if sig == signature:
            st.download_button(T["pdf_dl"], pdf_bytes, file_name=f"rapport_backtest_{pd.Timestamp('today'):%Y%m%d}.pdf",
                               mime="application/pdf", key="k_pdf_dl")
        else:
            st.warning(T["pdf_stale"])
