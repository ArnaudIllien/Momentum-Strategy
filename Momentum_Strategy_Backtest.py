import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tempfile, os, io

# ══════════════════════════════════════════════════════════════════════════════
# TRADUCTIONS
# ══════════════════════════════════════════════════════════════════════════════

LANG = {
    "fr": {
        "title":            "📈 Quant Strategy Dashboard",
        "expander_explain": "💡 Explication — comment fonctionne cette stratégie ?",
        "expander_limits":  "⚠️ Limites de la stratégie",
        "select_asset":     "Indice / Actif",
        "date_start":       "Date de début",
        "date_end":         "Date de fin",
        "date_error":       "La date de début doit être antérieure à la date de fin.",
        "caption_train":    "Train",
        "caption_test":     "Test",
        "params":           "Paramètres",
        "window_label":     "Window (momentum)",
        "fee_label":        "Transaction cost (par trade)",
        "perf":             "Performance",
        "sharpe_train":     "Sharpe — Train",
        "sharpe_oos":       "Sharpe — Out-of-sample",
        "equity_title":     "Equity curve",
        "equity_chart":     "Equity curve — train + out-of-sample",
        "equity_ylabel":    "Valeur du portefeuille (base 1)",
        "optim_title":      "Optimisation de la fenêtre",
        "optim_button":     "Lancer l'optimisation (train uniquement)",
        "optim_spinner":    "Calcul en cours...",
        "best_window":      "Meilleure fenêtre",
        "sharpe_optimal":   "Sharpe train optimal",
        "sharpe_oos2":      "Sharpe Out Of Sample",
        "optim_chart":      "Sharpe ratio vs fenêtre de momentum (train)",
        "buy_hold":         "Buy & Hold",
        "strategy":         "Stratégie",
        "train_test_label": "Train / Test",
        "risk_title":       "Métriques de risque",
        "max_dd":           "Max Drawdown",
        "calmar":           "Calmar Ratio",
        "win_rate":         "Win Rate",
        "ann_vol":          "Volatilité annualisée",
        "ann_ret":          "Rendement annualisé",
        "dd_chart":         "Drawdown de la stratégie",
        "multi_title":      "Tableau de bord multi-actifs",
        "multi_spinner":    "Analyse de tous les actifs...",
        "multi_button":     "Lancer l'analyse multi-actifs",
        "multi_sharpe":     "Sharpe OOS",
        "multi_dd":         "Max DD",
        "multi_calmar":     "Calmar",
        "multi_winrate":    "Win Rate",
        "multi_vol":        "Vol ann.",
        "multi_ret":        "Ret ann.",
        "pdf_button":       "📄 Télécharger le rapport PDF",
        "pdf_title":        "Rapport Quant Strategy",
        "pdf_spinner":      "Génération du PDF...",
        "explain_md": """
### L'idée de base : le momentum

Les marchés ont tendance à **continuer dans leur direction récente**.
Si un actif a monté ces derniers jours, on parie qu'il va continuer à monter.
C'est ce qu'on appelle l'effet *momentum*.

---

### Ce que fait la stratégie, pas à pas

| Étape | Ce qui se passe |
|-------|----------------|
| 1 | On calcule le **rendement quotidien** de l'actif (ex. +1.2 % aujourd'hui) |
| 2 | On **somme** ces rendements sur les `N` derniers jours — c'est la *fenêtre* |
| 3 | Si la somme est **positive** → on achète (+1). Si **négative** → on vend à découvert (−1) |
| 4 | On décale la position d'**un jour** : on ne peut pas trader dans le futur |
| 5 | On retire les **frais de transaction** à chaque changement de position |

---

### Les indicateurs clés

**Sharpe ratio** — mesure si la stratégie rapporte *vraiment* ou juste par chance.
- **> 1** : bon · **entre 0 et 1** : acceptable · **< 0** : perd de l'argent

**Buy & Hold** — acheter et ne jamais vendre. Si la stratégie ne le bat pas, elle ne sert à rien.

**Max Drawdown** — perte maximale entre un pic et le creux suivant.
- **> −10%** : stable · **−10% à −30%** : acceptable · **< −30%** : dangereux

**Calmar Ratio** — rendement annualisé divisé par le Max Drawdown en valeur absolue.
- **> 1** : le gain compense le risque · **< 0.5** : le drawdown est trop coûteux

**Win Rate** — part des jours où la stratégie est gagnante.
- À lire avec le Calmar : un Win Rate de 40% peut rester rentable si les jours gagnants rapportent bien plus que les jours perdants.

**Volatilité annualisée** — dispersion des rendements quotidiens extrapolée sur un an.
- **< 10%** : stable · **15–25%** : comparable à un indice actions · **> 40%** : très risqué

**Rendement annualisé** — moyenne des rendements quotidiens nets × 252.
- Toujours à lire avec la volatilité : 20% de rendement avec 40% de vol est moins solide que 10% avec 8% de vol.
---

### Train vs Test — éviter de se mentir

- **Train (70 %)** : on optimise ici
- **Test (30 %)** : on vérifie sur des données *jamais vues*

Si le Sharpe s'effondre sur le test → **overfitting**.
""",
        "limits_md": """
### Ce que le modèle ne capture pas

#### 1. Slippage
Exécution parfaite au prix de clôture supposée — irréaliste en pratique.

#### 2. Frais symétriques et constants
Pas de spread bid/ask ni de market impact modélisé.

#### 3. Prix de clôture non tradable
Look-ahead bias partiel : le signal est connu après la clôture.

#### 4. Short selling idéal
Vente à découvert supposée libre, sans coût d'emprunt.

#### 5. Pas de gestion du risque
Toujours 100 % investi — pas de stop-loss ni de position sizing.

#### 6. Overfitting de la fenêtre
50 fenêtres testées = risque de trouver la meilleure par chance.

#### 7. Un seul actif, une seule stratégie
Le momentum ne fonctionne pas en marché sans tendance.
""",
    },

    "en": {
        "title":            "📈 Quant Strategy Dashboard",
        "expander_explain": "💡 Explanation — how does this strategy work?",
        "expander_limits":  "⚠️ Strategy limitations",
        "select_asset":     "Index / Asset",
        "date_start":       "Start date",
        "date_end":         "End date",
        "date_error":       "Start date must be before end date.",
        "caption_train":    "Train",
        "caption_test":     "Test",
        "params":           "Parameters",
        "window_label":     "Window (momentum)",
        "fee_label":        "Transaction cost (per trade)",
        "perf":             "Performance",
        "sharpe_train":     "Sharpe — Train",
        "sharpe_oos":       "Sharpe — Out-of-sample",
        "equity_title":     "Equity curve",
        "equity_chart":     "Equity curve — train + out-of-sample",
        "equity_ylabel":    "Portfolio value (base 1)",
        "optim_title":      "Window optimisation",
        "optim_button":     "Run optimisation (train only)",
        "optim_spinner":    "Computing...",
        "best_window":      "Best window",
        "sharpe_optimal":   "Best train Sharpe",
        "sharpe_oos2":      "Out-of-sample Sharpe",
        "optim_chart":      "Sharpe ratio vs momentum window (train)",
        "buy_hold":         "Buy & Hold",
        "strategy":         "Strategy",
        "train_test_label": "Train / Test",
        "risk_title":       "Risk metrics",
        "max_dd":           "Max Drawdown",
        "calmar":           "Calmar Ratio",
        "win_rate":         "Win Rate",
        "ann_vol":          "Annualised Vol",
        "ann_ret":          "Annualised Return",
        "dd_chart":         "Strategy Drawdown",
        "multi_title":      "Multi-asset dashboard",
        "multi_spinner":    "Analysing all assets...",
        "multi_button":     "Run multi-asset analysis",
        "multi_sharpe":     "Sharpe OOS",
        "multi_dd":         "Max DD",
        "multi_calmar":     "Calmar",
        "multi_winrate":    "Win Rate",
        "multi_vol":        "Ann. Vol",
        "multi_ret":        "Ann. Ret",
        "pdf_button":       "📄 Download PDF report",
        "pdf_title":        "Quant Strategy Report",
        "pdf_spinner":      "Generating PDF...",
        "explain_md": """
### Core idea: momentum

Markets tend to **continue in their recent direction**.
If an asset has risen over the past few days, we bet it will keep rising.
This is called the *momentum* effect.

---

### Strategy step by step

| Step | What happens |
|------|-------------|
| 1 | Compute the asset's **daily return** (e.g. +1.2% today) |
| 2 | **Sum** those returns over the last `N` days — the *window* |
| 3 | Positive sum → buy (+1). Negative sum → short sell (−1) |
| 4 | Shift position by **one day**: we can't trade in the past |
| 5 | Subtract **transaction costs** on every position change |

---

### Key metrics

**Sharpe ratio** — measures whether the strategy earns returns consistently.
- **> 1**: good · **0–1**: acceptable · **< 0**: losing money

**Buy & Hold** — buy and never sell. If the strategy doesn't beat it, it's useless.

**Max Drawdown** — largest peak-to-trough loss over the period.
- **> −10%**: stable · **−10% to −30%**: acceptable · **< −30%**: dangerous

**Calmar Ratio** — annualised return divided by the absolute Max Drawdown.
- **> 1**: gain outweighs risk · **< 0.5**: drawdown is too costly relative to return

**Win Rate** — share of days where the strategy is profitable.
- Read alongside Calmar: a 40% Win Rate can still be profitable if winning days significantly outweigh losing ones.

**Annualised Volatility** — dispersion of daily returns scaled to a full year.
- **< 10%**: stable · **15–25%**: comparable to an equity index · **> 40%**: highly risky

**Annualised Return** — average daily net return × 252.
- Always read with volatility: 20% return with 40% vol is weaker than 10% return with 8% vol
---

### Train vs Test — no cheating

- **Train (70%)**: optimise here
- **Test (30%)**: validate on *unseen* data

If the Sharpe collapses on the test set → **overfitting**.
""",
        "limits_md": """
### What the model doesn't capture

#### 1. Slippage
Perfect execution at closing price assumed — unrealistic in practice.

#### 2. Symmetric, constant fees
No bid/ask spread or market impact modelled.

#### 3. Close price not tradable
Partial look-ahead bias: signal is known only after the close.

#### 4. Ideal short selling
Frictionless shorting assumed, with no borrowing cost.

#### 5. No risk management
Always 100% invested — no stop-loss or position sizing.

#### 6. Window overfitting
50 windows tested = high chance of finding the best one by luck.

#### 7. Single asset, single strategy
Momentum fails in ranging markets.
""",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# LOGIQUE QUANT
# ══════════════════════════════════════════════════════════════════════════════

def strategy_fn(df, window=1, fee=0.0001):
    df = df.copy()
    df['ret']      = np.log(df.Close.pct_change() + 1)
    df['prior_n']  = df.ret.rolling(window).sum()
    df['position'] = [1 if i > 0 else -1 for i in df.prior_n]
    df['trade']    = df['position'].diff().abs()
    df['costs']    = df['trade'] * fee
    df['strat']    = df.position.shift(1) * df['ret']
    df['strat_net']= df['strat'] - df['costs']
    df['equity_ret']= df['strat_net'].cumsum()
    df.dropna(inplace=True)
    return df

def sharpe_ratio(df, rf=0.0):
    r = df['strat_net'].dropna()
    if r.std() == 0:
        return 0.0
    return np.sqrt(252) * (r.mean() - rf) / r.std()

def risk_metrics(df):
    r = df['strat_net'].dropna()
    equity = np.exp(r.cumsum())

    # Max Drawdown
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max
    max_dd   = dd.min()

    # Calmar
    ann_ret = r.mean() * 252
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    # Win rate
    win_rate = (r > 0).mean()

    # Vol annualisée
    ann_vol = r.std() * np.sqrt(252)

    return {
        "max_dd"  : max_dd,
        "calmar"  : calmar,
        "win_rate": win_rate,
        "ann_vol" : ann_vol,
        "ann_ret" : ann_ret,
        "dd_series": dd,
    }

def optimize_window(df, windows=range(1, 51), fee=0.0001):
    results = []
    for w in windows:
        df_temp = strategy_fn(df, window=w, fee=fee)
        results.append((w, sharpe_ratio(df_temp)))
    return results

def equity_curve(df_strat):
    return np.exp(df_strat[['ret', 'strat_net']].cumsum())

# ══════════════════════════════════════════════════════════════════════════════
# TICKERS
# ══════════════════════════════════════════════════════════════════════════════

TICKERS = {
    "EUR/USD" : "EURUSD=X",
    "BTC/USD" : "BTC-USD",
    "S&P 500" : "^GSPC",
    "Gold"    : "GC=F",
    "Apple"   : "AAPL",
    "Nasdaq"  : "^IXIC",
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — LANGUE
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    en_mode = st.toggle("🇬🇧 English", value=False)

lang = "en" if en_mode else "fr"
T    = LANG[lang]

# ══════════════════════════════════════════════════════════════════════════════
# UI PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

st.title(T["title"])

with st.expander(T["expander_explain"], expanded=False):
    st.markdown(T["explain_md"])

with st.expander(T["expander_limits"], expanded=False):
    st.markdown(T["limits_md"])

# ── Sélection actif + période ──────────────────────────────────────────────

ticker_label = st.selectbox(T["select_asset"], options=list(TICKERS.keys()))
ticker       = TICKERS[ticker_label]

col_d1, col_d2 = st.columns(2)
start_date = col_d1.date_input(T["date_start"], value=pd.Timestamp("2020-08-01"))
end_date   = col_d2.date_input(T["date_end"],   value=pd.Timestamp("today"))

if start_date >= end_date:
    st.error(T["date_error"])
    st.stop()

# ── Chargement données ─────────────────────────────────────────────────────

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        st.error(f"Cannot load data for {ticker}.")
        st.stop()
    return df

df = load_data(ticker, start=start_date, end=end_date)

split    = int(len(df) * 0.7)
df_train = df.iloc[:split]
df_test  = df.iloc[split:]

col1, col2 = st.columns(2)
col1.caption(f"{T['caption_train']} : {df_train.index[0].date()} → {df_train.index[-1].date()}")
col2.caption(f"{T['caption_test']}  : {df_test.index[0].date()}  → {df_test.index[-1].date()}")

# ── Paramètres ─────────────────────────────────────────────────────────────

st.subheader(T["params"])
window = st.slider(T["window_label"], 1, 50, 10)
fee    = st.slider(T["fee_label"], min_value=0.0, max_value=0.001,
                   value=0.0001, step=0.00001, format="%.5f")

# ── Calcul stratégie ───────────────────────────────────────────────────────

df_full_strat  = strategy_fn(df, window=window, fee=fee)
split_idx      = df_full_strat.index.get_loc(df_train.index[-1])
df_train_strat = df_full_strat.iloc[:split_idx + 1]
df_test_strat  = df_full_strat.iloc[split_idx + 1:]

sr_train = sharpe_ratio(df_train_strat)
sr_test  = sharpe_ratio(df_test_strat)
metrics  = risk_metrics(df_test_strat)   # métriques sur OOS uniquement

# ── Performance Sharpe ─────────────────────────────────────────────────────

st.subheader(T["perf"])
m1, m2 = st.columns(2)
m1.metric(T["sharpe_train"], round(sr_train, 3))
m2.metric(T["sharpe_oos"],   round(sr_test,  3),
          delta=round(sr_test - sr_train, 3))

# ── Métriques de risque ────────────────────────────────────────────────────

st.subheader(T["risk_title"])
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric(T["max_dd"],   f"{metrics['max_dd']*100:.1f}%")
r2.metric(T["calmar"],   f"{metrics['calmar']:.2f}"
          if not np.isnan(metrics['calmar']) else "N/A")
r3.metric(T["win_rate"], f"{metrics['win_rate']*100:.1f}%")
r4.metric(T["ann_vol"],  f"{metrics['ann_vol']*100:.1f}%")
r5.metric(T["ann_ret"],  f"{metrics['ann_ret']*100:.1f}%")

# ── Equity curve ───────────────────────────────────────────────────────────

st.subheader(T["equity_title"])

eq_train    = equity_curve(df_train_strat)
eq_test     = equity_curve(df_test_strat)
offset      = eq_train.iloc[-1] - eq_test.iloc[0]
eq_test_adj = eq_test + offset
equity_full = pd.concat([eq_train, eq_test_adj])
equity_full.columns = [T["buy_hold"], T["strategy"]]

fig_equity, ax = plt.subplots(figsize=(10, 4))
equity_full[T["buy_hold"]].plot(ax=ax, label=T["buy_hold"], color='steelblue',  linewidth=1)
equity_full[T["strategy"]].plot(ax=ax, label=T["strategy"], color='darkorange', linewidth=1)
ax.axvline(df_train_strat.index[-1], color='gray', linestyle='--',
           linewidth=0.8, label=T["train_test_label"])
ax.legend(fontsize=9)
ax.set_title(T["equity_chart"], fontsize=11)
ax.set_ylabel(T["equity_ylabel"])
st.pyplot(fig_equity)


# ── Optimisation fenêtre ───────────────────────────────────────────────────

st.subheader(T["optim_title"])

if st.button(T["optim_button"]):
    with st.spinner(T["optim_spinner"]):
        results = optimize_window(df_train_strat, windows=range(1, 51), fee=fee)

    best_window = max(results, key=lambda x: x[1])[0]
    best_sharpe = max(results, key=lambda x: x[1])[1]
    df_best_oos = strategy_fn(df_test_strat, window=best_window, fee=fee)
    oos_sharpe  = sharpe_ratio(df_best_oos)

    b1, b2, b3 = st.columns(3)
    b1.metric(T["best_window"],    best_window)
    b2.metric(T["sharpe_optimal"], round(best_sharpe, 3))
    b3.metric(T["sharpe_oos2"],    round(oos_sharpe,  3),
              delta=round(oos_sharpe - best_sharpe, 3))

    windows_list = [w for w, s in results]
    sharpes_list = [s for w, s in results]

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(windows_list, sharpes_list, color='steelblue', linewidth=1.2)
    ax3.axvline(best_window, color='darkorange', linestyle='--',
                linewidth=1, label=f'Best = {best_window}')
    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.set_xlabel("Window")
    ax3.set_ylabel("Sharpe ratio")
    ax3.set_title(T["optim_chart"], fontsize=11)
    ax3.legend(fontsize=9)
    st.pyplot(fig3)
    plt.close(fig3)
