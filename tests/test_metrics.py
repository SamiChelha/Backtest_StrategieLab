"""
Unit Tests — tests/test_metrics.py
====================================
Tests for the strategy_lab.metrics module.

Run:
    uv run pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from strategy_lab.metrics import (
    annualized_return,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    beta,
    information_ratio,
    compute_summary,
)


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------

def test_annualized_return_positive():
    """Constant positive monthly returns should produce a positive CAGR."""
    r = pd.Series([0.01] * 12)
    result = annualized_return(r, periods_per_year=12)
    assert result > 0, "CAGR must be positive for positive returns"


def test_annualized_return_zero():
    """Zero returns → CAGR = 0."""
    r = pd.Series([0.0] * 12)
    assert abs(annualized_return(r, periods_per_year=12)) < 1e-9


def test_annualized_return_empty():
    """Empty series → nan, no crash."""
    assert np.isnan(annualized_return(pd.Series([], dtype=float)))


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

def test_sharpe_ratio_positive():
    """Positive mean returns with variance → positive Sharpe."""
    r = pd.Series([0.01, 0.02, 0.015, 0.01, 0.03, 0.01] * 4)
    assert sharpe_ratio(r, periods_per_year=12) > 0


def test_sharpe_ratio_zero_variance():
    """Zero-variance series → nan (previously was 0.0, now fixed)."""
    r = pd.Series([0.0] * 12)
    result = sharpe_ratio(r, periods_per_year=12)
    assert np.isnan(result), f"Expected nan, got {result}"


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

def test_sortino_ratio_positive():
    """Mixed returns with some negative → finite Sortino."""
    r = pd.Series([0.02, -0.01, 0.015, -0.005, 0.03, 0.01] * 4)
    result = sortino_ratio(r, periods_per_year=12)
    assert np.isfinite(result)


def test_sortino_ratio_no_downside():
    """No negative returns → nan (undefined downside risk, not 0)."""
    r = pd.Series([0.01, 0.02, 0.03] * 4)
    result = sortino_ratio(r, periods_per_year=12)
    assert np.isnan(result), "Sortino must be nan when there is no downside risk"


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_known_value():
    """Simulate 1 → 2 → 1 wealth path: 50% drawdown from peak."""
    log_returns = pd.Series([np.log(2), np.log(0.5)])
    result = max_drawdown(log_returns)
    assert abs(result - (-0.5)) < 1e-6, f"Expected -0.5, got {result}"


def test_max_drawdown_no_drawdown():
    """Monotonically increasing returns → ~0 drawdown."""
    r = pd.Series([0.05] * 12)
    assert max_drawdown(r) >= -1e-9


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

def test_calmar_ratio_positive():
    """Positive returns with a drawdown → positive Calmar."""
    r = pd.Series([0.02, -0.01, 0.015, -0.005, 0.03, 0.01] * 4)
    result = calmar_ratio(r, periods_per_year=12)
    assert np.isfinite(result) and result > 0


def test_calmar_ratio_no_drawdown():
    """No drawdown → nan (undefined, not inf)."""
    r = pd.Series([0.01] * 12)
    result = calmar_ratio(r, periods_per_year=12)
    assert np.isnan(result), "Calmar must be nan when max_drawdown is 0"


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------

def test_beta_identical_series():
    """Strategy == benchmark → beta ≈ 1."""
    r = pd.Series([0.01, -0.02, 0.03, 0.01, -0.005] * 4)
    result = beta(r, r)
    assert abs(result - 1.0) < 1e-9, f"Expected 1.0, got {result}"


def test_beta_too_short():
    """Less than 2 aligned points → nan."""
    r1 = pd.Series([0.01], index=pd.to_datetime(["2020-01-01"]))
    r2 = pd.Series([0.02], index=pd.to_datetime(["2020-01-01"]))
    assert np.isnan(beta(r1, r2))


# ---------------------------------------------------------------------------
# information_ratio
# ---------------------------------------------------------------------------

def test_information_ratio_same_returns():
    """IR = 0 when strategy equals benchmark (no excess return)."""
    r = pd.Series([0.01, -0.01, 0.02] * 6)
    result = information_ratio(r, r, periods_per_year=12)
    assert np.isnan(result) or abs(result) < 1e-9


def test_information_ratio_outperformance():
    """IR > 0 when strategy consistently beats benchmark."""
    bench = pd.Series([0.005] * 24)
    strat = pd.Series([0.010] * 24)
    result = information_ratio(strat, bench, periods_per_year=12)
    assert np.isnan(result), "IR is nan when excess returns have zero variance"


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------

def test_compute_summary_keys():
    """compute_summary must return all expected keys."""
    r = pd.Series([0.01, -0.005, 0.02, 0.01, -0.01, 0.015] * 4)
    bench = pd.Series([0.008] * len(r))
    result = compute_summary(r, bench)
    expected_keys = {
        "annualized_return", "sharpe", "sortino", "max_drawdown",
        "calmar", "beta", "information_ratio", "total_return", "n_months"
    }
    assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"


def test_compute_summary_no_benchmark():
    """compute_summary without benchmark → beta and IR are nan."""
    r = pd.Series([0.01, -0.005, 0.02] * 4)
    result = compute_summary(r, benchmark_returns=None)
    assert np.isnan(result["beta"])
    assert np.isnan(result["information_ratio"])


def test_compute_summary_n_months():
    """n_months must equal the length of the input series."""
    r = pd.Series([0.01] * 20)
    result = compute_summary(r)
    assert result["n_months"] == 20
