# Strategy Lab — Walk-Forward Backtester

A rigorous **out-of-sample** portfolio backtesting framework built with Python and Streamlit.

## What this does

Strategy Lab tests portfolio construction strategies on historical price data using a
**walk-forward methodology with an expanding window**:

- The model is trained on all available history up to month `t-1`
- It predicts weights for month `t` — which it has never seen
- This repeats every month, so **every return in the results is strictly out-of-sample**

No look-ahead bias. No curve-fitting on the test set.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Getting started

```bash
# Install dependencies (first time only)
uv sync

# Launch the Streamlit app
uv run streamlit run app/main.py

# Or use the desktop launcher (macOS — starts the server and opens Chrome)
./launcher.sh
```

The app is served at http://127.0.0.1:8501.

## Run tests

```bash
uv run pytest tests/ -v
```

## Project structure

```
.
├── app/
│   └── main.py                    # Streamlit UI (no business logic here)
├── src/strategy_lab/
│   ├── __init__.py                # Public API
│   ├── data.py                    # Download prices, compute log-returns
│   ├── engine.py                  # Walk-forward engine + all strategies
│   ├── metrics.py                 # Sharpe, Sortino, Calmar, IR, etc.
│   └── black_litterman.py         # Black-Litterman allocation
├── tests/
│   ├── test_engine.py             # Walk-forward correctness tests
│   ├── test_metrics.py            # Metrics formula tests
│   └── test_black_litterman.py    # Black-Litterman tests
├── reference/
│   └── thesis_code.py             # Frozen standalone version used for the thesis
├── pyproject.toml                 # Dependencies & build config
├── uv.lock                        # Pinned dependency versions
└── launcher.sh                    # Desktop launcher (Chrome app mode)
```

The package lives under `src/` and is imported as `strategy_lab`. The UI in `app/`
imports from it and holds no business logic of its own — that separation is what
makes the engine testable independently of Streamlit.

## Available strategies

| Strategy | Description |
|---|---|
| Equal Weight | 1/N allocation — rebalanced periodically |
| Minimum Variance | Minimum portfolio variance (long-only, SLSQP) |
| Rolling Sharpe | Weight proportional to Sharpe ratio over the training window |
| Momentum (12-1) | Cross-sectional momentum, skipping the last month |
| Robust Rank Momentum | Like Momentum, but ranks instead of raw returns |
| Black-Litterman | Market equilibrium prior blended with investor views |

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| Min. training months | 12 | Warm-up period before the first OOS prediction |
| Rebalance every N months | 1 | Trading frequency |
| Transaction cost (bps) | 10 | Applied to turnover at each rebalance |
| Window type | Expanding | Expanding (rigorous) or Rolling (fixed lookback) |

## Disclaimer

This is a research and educational tool. Backtested results are not indicative of
future performance and nothing here constitutes investment advice.
