# Momentum Strategy Backtester

> A multi-asset, cross-sectional momentum backtester with a Streamlit dashboard: flexible position sizing (cash earns the risk-free rate), asset-class caps, a drawdown circuit breaker, walk-forward optimisation, a head-to-head comparison against classic allocations, and a one-click PDF report.

---

## Strategy

**Cross-sectional momentum** ranks assets *against each other* instead of asking whether a single asset is trending.

At each rebalance date:

1. **Score** every asset: return over the last `lookback` days, skipping the most recent `skip` days (short-term reversal). Optionally divide by realised volatility, so assets are compared at equal risk.
2. **Select** the top `N` assets (long) and, optionally, the bottom `M` (short).
3. **Weight** each leg: equal weight, inverse volatility, or risk parity (equal risk contribution).
4. **Scale** exposure:
   - invested share below 100%;
   - trend filter (only buy an asset whose own momentum is positive; otherwise its slot stays in cash);
   - volatility target and leverage cap;
   - **asset-class caps** (default 60% equities / 20% bonds / 10% commodities / 10% crypto & other).
5. **Hold cash** for everything not invested. Cash earns the **risk-free rate** (US 13-week T-bill, `^IRX`).

### Overlays

| Overlay | Behaviour |
|---|---|
| Asset-class caps | Long weight per class is capped. The excess is redistributed to other selected assets whose class has room, otherwise it stays in cash. |
| Drawdown circuit breaker | When the drawdown from the peak exceeds a threshold, exposure is cut immediately (even between rebalances), then restored after a cooldown. |
| Volatility target | Exposure is scaled so the ex-ante portfolio volatility matches a target. |

---

## Dashboard

| Tab | Content |
|---|---|
| **Performance** | Equity vs equal-weight benchmark and cash, drawdown, rolling Sharpe, full train/test metric table |
| **Comparison** | Cross-sectional momentum vs the same without caps, time-series momentum, strategic allocation (60/20/10/10), equal weight and risk parity. Same assets, fees, rebalancing and start date for all. |
| **Allocation & cash** | Weights over time, gross/net exposure and cash, exposure by asset class, current target weights |
| **Correlations & risk** | Correlation matrix (full / train / test), rolling average correlation, risk contributions, diversification ratio |
| **Per asset** | Return, volatility, Sharpe, time held, P&L contribution |
| **Robustness** | Lookback × number-of-assets grid (train vs test Sharpe), fee sensitivity |
| **Walk-forward** | Periodic re-optimisation on past data only. Shows the stitched out-of-sample curve and the Sharpe each choice *promised* vs what it *delivered*. |
| **Data** | CSV exports and a **PDF report** with the same charts |

**Universe:** about 40 built-in assets across equities, equity ETFs, bonds, commodities and other (indices, FX, Bitcoin). **Any Yahoo Finance asset** can be added through a name search. Non-USD assets are converted to USD automatically.

The interface is bilingual (French / English).

---

## Methodology (no look-ahead)

The engine is weight-based. For each day `t`:

1. The portfolio held since the previous close earns
   `r_t = Σ wᵢ·Rᵢ,t + cash·rf_t − borrow·Σ|shorts|`, using **simple** returns, which aggregate correctly across assets.
2. Weights drift with prices.
3. On rebalance days, the signal uses data up to the close of `t`. Trades are charged on the actual turnover: `fee · Σ|w_target − w_drifted|`.
4. New weights only earn from `t+1`.

Train/test metrics come from **one** backtest over the full history, split by date, so the test period has no cold start.

All assets are aligned on a **business-day calendar**: crypto weekend moves are included in Monday's return.

---

## Run locally

```bash
git clone https://github.com/ArnaudIllien/Momentum-Strategy.git
cd Momentum-Strategy
pip install -r requirements.txt
streamlit run Momentum_Strategy_Backtest.py
```

The dashboard opens at <http://localhost:8501>.

PDF export uses `kaleido`, which needs a local Chrome or Chromium installation.

### Tests

```bash
python -m tests.test_engine
```

The tests run offline on synthetic data. They cover:
- no look-ahead, for the engine and for the walk-forward;
- cash accrual;
- transaction costs;
- leverage and volatility caps;
- risk-parity convergence, including with negatively correlated assets;
- class caps;
- the circuit breaker;
- calendar alignment.

---

## Project structure

```
├── Momentum_Strategy_Backtest.py   # Streamlit dashboard
├── backtester/
│   ├── data.py          # Yahoo download, business-day alignment, USD conversion, risk-free rate, search
│   ├── signals.py       # momentum scores
│   ├── portfolio.py     # selection, weighting (equal / inverse vol / risk parity), caps, sizing
│   ├── engine.py        # daily weight-based backtest, costs, circuit breaker
│   ├── metrics.py       # Sharpe, Sortino, drawdown, beta/alpha, risk contributions…
│   ├── strategies.py    # comparison strategies
│   ├── walkforward.py   # walk-forward optimisation
│   ├── universe.py      # built-in asset universe and asset classes
│   └── report.py        # PDF report (Plotly → PNG via kaleido, layout via fpdf2)
├── tests/test_engine.py
└── requirements.txt
```

---

## Limitations

- **Execution at the close:** the signal and the trade use the same closing price, which is optimistic.
- **Linear fees:** no variable spread and no market impact.
- **Continuous futures (`=F`):** roll yield and roll cost are ignored.
- **Survivorship bias:** the universe is chosen today.
- **Overfitting:** every grid search finds *something*. The walk-forward tab shows how much of the in-sample Sharpe survives out-of-sample.
