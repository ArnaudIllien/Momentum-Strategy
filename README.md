# Momentum Strategy Backtester

> A systematic, data-driven momentum trading strategy built in Python — featuring a live Streamlit dashboard with train/test separation, walk-forward optimisation, and full risk analytics.

---

## Strategy Overview

The strategy exploits the **momentum effect**: assets that have risen over a recent window tend to continue rising. The signal is purely systematic — no discretionary input.

**Signal logic:**
- Compute the rolling sum of daily log-returns over a configurable window `N`
- If the sum is **positive** → Long (+1); if **negative** → Short (−1)
- Position is shifted by one day to avoid look-ahead bias
- Transaction costs are subtracted on every position change

**Train / Test split:**
- 70% of data → **in-sample optimisation**
- 30% → **out-of-sample evaluation** (never seen during optimisation)

The Sharpe delta between train and test is displayed explicitly to flag overfitting.

---

## Features

| Feature | Description |
|---|---|
| Equity curve | Strategy vs Buy & Hold, with train/test separator |
| Walk-forward optimisation | Sharpe ratio tested across 50 momentum windows (train only) |
| Risk metrics | Max Drawdown, Calmar Ratio, Win Rate, Annualised Vol & Return |
| Multi-asset dashboard | One-click comparison across 6 assets |
| Bilingual UI | French / English toggle |
| Live data | Real-time prices pulled from Yahoo Finance via `yfinance` |

**Assets covered:** S&P 500 · Nasdaq · EUR/USD · BTC/USD · Gold · Apple

---

## Risk & Limitations

This backtester is intentionally transparent about its assumptions:

- **No slippage** — execution assumed at close price
- **Constant, symmetric fees** — no bid/ask spread or market impact
- **Ideal short selling** — no borrow cost modelled
- **No position sizing** — always 100% invested
- **Window overfitting risk** — testing 50 windows increases the chance of spurious best results
- **Single factor** — momentum underperforms in range-bound or mean-reverting markets

These limitations are documented directly in the dashboard UI.

---

## Tech Stack

- **Python 3.11+**
- `streamlit` — interactive dashboard
- `pandas` / `numpy` — data pipeline and signal computation
- `matplotlib` — equity curve, drawdown, optimisation charts
- `yfinance` — market data

---

## Run Locally

```bash
git clone https://github.com/ArnaudIllien/momentum-strategy-backtester.git
cd momentum-strategy-backtester
pip install -r requirements.txt
streamlit run Momentum_Strategy_Backtest.py
```

**requirements.txt**
```
streamlit
yfinance
pandas
numpy
matplotlib
```

---

## Project Structure

```
momentum-strategy-backtester/
│
├── Momentum_Strategy_Backtest.py   # Main app (strategy + Streamlit UI)
├── requirements.txt
└── README.md
```

---

## Purpose

```
Built as a personal project in quantitative finance. The objective was to develop a rigorous systematic strategy pipeline, from raw data ingestion to out-of-sample risk reporting — reflecting the standards used in professional quantitative research.
---
```

## Roadmap

- [ ] Add Bollinger Band mean-reversion strategy for comparison
- [ ] Implement proper walk-forward cross-validation (rolling windows)
- [ ] Add position sizing (volatility targeting / Kelly criterion)
- [ ] Integrate transaction cost sensitivity analysis

---
