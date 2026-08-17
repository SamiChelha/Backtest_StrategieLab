"""
Analytics & Metrics Module — strategy_lab.metrics
===================================================
Performance metrics for backtested strategies.

Public API
----------
- annualized_return()
- sharpe_ratio()
- sortino_ratio()
- max_drawdown()
- calmar_ratio()
- beta()
- information_ratio()
- compute_summary()
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _excess(log_returns: pd.Series, rf: pd.Series | None) -> pd.Series:
    """Return excess returns = log_returns − rf, aligned by index.

    If ``rf`` is None or empty, returns ``log_returns`` unchanged (rf = 0).
    Only months present in both series are used for the subtraction;
    months missing from rf fall back to 0 for that period.
    """
    if rf is None or rf.empty:
        return log_returns
    rf_aligned = rf.reindex(log_returns.index).fillna(0.0)
    return log_returns - rf_aligned


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def annualized_return(log_returns: pd.Series, periods_per_year: int = 12) -> float:

    """
    Compound Annual Growth Rate (CAGR) from a series of log-returns.

    Formula: exp(mean(r) * N) - 1
    where N = periods per year, r = periodic log-returns.

    Returns a fraction (e.g. 0.08 = 8% per year).
    Comparable across backtests of different lengths.
    """
    if len(log_returns) == 0:
        return float("nan")
    return float(np.exp(log_returns.mean() * periods_per_year) - 1)


def sharpe_ratio(
    log_returns: pd.Series,
    rf: pd.Series | None = None,
    periods_per_year: int = 12,
) -> float:
    """
    Annualized Sharpe ratio.

    Formula: sqrt(N) * mean(excess_r) / std(excess_r)
    where excess_r = log_returns - rf (or log_returns if rf is None/0).

    Parameters
    ----------
    rf : pd.Series or None
        Monthly risk-free log-returns. Aligned by index. None → rf = 0.
    """
    if len(log_returns) == 0:
        return float("nan")
    excess = _excess(log_returns, rf)
    if excess.std() == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def sortino_ratio(
    log_returns: pd.Series,
    rf: pd.Series | None = None,
    periods_per_year: int = 12,
) -> float:
    """
    Annualized Sortino ratio.

    Like Sharpe, but penalises ONLY downside volatility of excess returns.
    A higher Sortino vs Sharpe means the strategy's volatility is mostly upside.

    Parameters
    ----------
    rf : pd.Series or None
        Monthly risk-free log-returns. Aligned by index. None → rf = 0.
    """
    if len(log_returns) == 0:
        return float("nan")
    excess = _excess(log_returns, rf)
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std())


def max_drawdown(log_returns: pd.Series) -> float:
    """
    Maximum Peak-to-Trough drawdown expressed as a negative fraction.

    Example: -0.35 means the portfolio fell 35% from its highest point.
    """
    if len(log_returns) == 0:
        return float("nan")
    wealth = np.exp(log_returns.cumsum())
    peaks = wealth.cummax()
    dd = (wealth - peaks) / peaks
    return float(dd.min())


def calmar_ratio(log_returns: pd.Series, periods_per_year: int = 12) -> float:
    """
    Calmar ratio: annualized return divided by maximum drawdown (absolute value).

    Measures how much return you earn per unit of worst historic pain.
    Higher is better. Infinite if there is no drawdown.
    """
    ann_ret = annualized_return(log_returns, periods_per_year)
    mdd = max_drawdown(log_returns)
    if np.isnan(mdd) or mdd == 0:
        return float("nan")
    return float(ann_ret / abs(mdd))


def beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Beta of the strategy relative to a benchmark.

    Measures market sensitivity:
      - Beta = 1.0  → moves exactly with the benchmark
      - Beta > 1.0  → amplifies market moves (aggressive)
      - Beta < 1.0  → less sensitive than benchmark (defensive)
      - Beta < 0    → moves opposite to benchmark (hedge)

    Formula: cov(strategy, benchmark) / var(benchmark)
    """
    aligned = pd.concat(
        [strategy_returns, benchmark_returns], axis=1, join="inner"
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    strat = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]
    var_bench = bench.var()
    if var_bench == 0:
        return float("nan")
    return float(strat.cov(bench) / var_bench)


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """
    Annualized Information Ratio (IR).

    Measures the quality of active returns (alpha) relative to tracking error.

    Formula: sqrt(N) * mean(excess_returns) / std(excess_returns)
    where excess_returns = strategy_returns - benchmark_returns.

    Interpretation:
      - IR > 0.5  → good active management
      - IR > 1.0  → excellent active management
      - IR < 0    → underperforming the benchmark on a risk-adjusted basis
    """
    aligned = pd.concat(
        [strategy_returns, benchmark_returns], axis=1, join="inner"
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    if excess.std() == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


# ---------------------------------------------------------------------------
# Centralised summary
# ---------------------------------------------------------------------------

def compute_summary(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    rf: pd.Series | None = None,
    periods_per_year: int = 12,
) -> dict:
    """
    Compute all performance metrics in one call.

    Parameters
    ----------
    strategy_returns : pd.Series
        Monthly log-returns of the strategy (Out-of-Sample walk-forward).
    benchmark_returns : pd.Series or None
        Monthly log-returns of the benchmark (e.g. SPY).
        If None, Beta and Information Ratio will be NaN.
    rf : pd.Series or None
        Monthly risk-free log-returns used in Sharpe & Sortino.
        If None, risk-free rate is assumed to be 0.
    periods_per_year : int
        Number of periods per year (12 for monthly data).

    Returns
    -------
    dict
        Keys: 'annualized_return', 'sharpe', 'sortino', 'max_drawdown',
              'calmar', 'beta', 'information_ratio', 'total_return',
              'n_months'
    """
    wealth = np.exp(strategy_returns.cumsum())
    total_ret = float(wealth.iloc[-1] - 1.0) if len(wealth) > 0 else float("nan")

    b  = beta(strategy_returns, benchmark_returns) if benchmark_returns is not None else float("nan")
    ir = information_ratio(strategy_returns, benchmark_returns, periods_per_year) if benchmark_returns is not None else float("nan")

    return {
        "annualized_return": annualized_return(strategy_returns, periods_per_year),
        "sharpe":            sharpe_ratio(strategy_returns, rf=rf, periods_per_year=periods_per_year),
        "sortino":           sortino_ratio(strategy_returns, rf=rf, periods_per_year=periods_per_year),
        "max_drawdown":      max_drawdown(strategy_returns),
        "calmar":            calmar_ratio(strategy_returns, periods_per_year),
        "beta":              b,
        "information_ratio": ir,
        "total_return":      total_ret,
        "n_months":          len(strategy_returns),
    }
