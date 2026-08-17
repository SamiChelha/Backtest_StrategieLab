"""
Unit Tests — tests/test_engine.py
====================================
Tests for the strategy_lab.engine module.

Run:
    uv run pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from strategy_lab.engine import run_walk_forward, STRATEGIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n_months: int = 60, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic monthly log-returns for testing."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.005, 0.03, size=(n_months, n_assets))
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=cols)


# ---------------------------------------------------------------------------
# Basic smoke tests (all strategies, both methods)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", list(STRATEGIES.keys()))
@pytest.mark.parametrize("method", ["expanding", "rolling"])
def test_run_walk_forward_smoke(strategy, method):
    """All strategy/method combinations must run without error."""
    returns = _make_returns()
    results, folds_meta = run_walk_forward(
        returns, strategy=strategy, min_train_months=12, method=method
    )
    assert not results.empty, f"{strategy}/{method} returned empty results"
    assert "Strategy" in results.columns
    assert not folds_meta.empty


# ---------------------------------------------------------------------------
# Expanding window correctness
# ---------------------------------------------------------------------------

def test_expanding_first_oos_date():
    """First OOS date must be exactly index[min_train_months]."""
    returns = _make_returns(n_months=36)
    min_train = 12
    results, folds_meta = run_walk_forward(returns, min_train_months=min_train, method="expanding")
    expected_first = returns.index[min_train]
    actual_first = results.index[0]
    assert actual_first == expected_first, (
        f"Expected first OOS date {expected_first}, got {actual_first}"
    )


def test_expanding_train_end_before_test_date():
    """For every fold, train_end must be strictly before test_date."""
    returns = _make_returns(n_months=36)
    _, folds_meta = run_walk_forward(returns, min_train_months=12, method="expanding")
    violations = folds_meta[folds_meta["train_end"] >= folds_meta["test_date"]]
    assert violations.empty, (
        f"{len(violations)} folds have train_end >= test_date (look-ahead leak!)"
    )


def test_expanding_window_grows():
    """Training window length must increase by 1 each month (expanding)."""
    returns = _make_returns(n_months=36)
    _, folds_meta = run_walk_forward(returns, min_train_months=12, method="expanding")
    train_lengths = (folds_meta["train_end"] - folds_meta["train_start"]).dt.days
    # Each successive training window should be longer than the previous
    diffs = train_lengths.diff().dropna()
    assert (diffs > 0).all(), "Training window must strictly grow in expanding mode"


# ---------------------------------------------------------------------------
# Rolling window correctness
# ---------------------------------------------------------------------------

def test_rolling_window_fixed_length():
    """In rolling mode, every training window should span ~ min_train_months."""
    returns = _make_returns(n_months=48)
    min_train = 12
    _, folds_meta = run_walk_forward(returns, min_train_months=min_train, method="rolling")
    approx_days = min_train * 30
    train_days = (folds_meta["train_end"] - folds_meta["train_start"]).dt.days
    # Allow tolerance of ±35 days (month-end variation)
    assert (train_days > approx_days - 35).all()
    assert (train_days < approx_days + 35).all()


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------

def test_transaction_cost_reduces_return():
    """With costs > 0, total return must be lower than with costs = 0."""
    returns = _make_returns()
    results_free, _ = run_walk_forward(returns, transaction_cost_bps=0)
    results_cost, _ = run_walk_forward(returns, transaction_cost_bps=50)
    total_free = results_free["Strategy"].sum()
    total_cost = results_cost["Strategy"].sum()
    assert total_free > total_cost, "Transaction costs must reduce total return"


def test_zero_transaction_cost_no_penalty():
    """With costs = 0, folds should show cost_bps = 0 everywhere."""
    returns = _make_returns()
    _, folds_meta = run_walk_forward(returns, transaction_cost_bps=0)
    assert folds_meta["cost_bps"].sum() == 0.0


# ---------------------------------------------------------------------------
# Unknown strategy fallback
# ---------------------------------------------------------------------------

def test_unknown_strategy_fallback():
    """An unknown strategy name must fall back to Equal Weight silently."""
    returns = _make_returns()
    results_ew, _ = run_walk_forward(returns, strategy="Equal Weight")
    results_unk, _ = run_walk_forward(returns, strategy="NONEXISTENT_STRATEGY_XYZ")
    pd.testing.assert_frame_equal(results_ew, results_unk)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_returns():
    """Empty input must return two empty DataFrames."""
    empty = pd.DataFrame()
    results, folds_meta = run_walk_forward(empty)
    assert results.empty
    assert folds_meta.empty


def test_insufficient_data():
    """Fewer rows than min_train_months must return empty DataFrames."""
    returns = _make_returns(n_months=5)
    results, folds_meta = run_walk_forward(returns, min_train_months=12)
    assert results.empty


def test_rebalance_frequency():
    """Rebalancing every 3 months: only ~1/3 of folds should have rebalanced=True."""
    returns = _make_returns(n_months=60)
    _, folds_meta = run_walk_forward(returns, min_train_months=12, rebalance_months=3)
    rebal_count = folds_meta["rebalanced"].sum()
    total_folds = len(folds_meta)
    # Allow ±1 due to integer division boundary
    expected = total_folds // 3
    assert abs(rebal_count - expected) <= 1, (
        f"Expected ~{expected} rebalances, got {rebal_count}"
    )


def test_output_shape():
    """Results columns must be 'Strategy' + one per asset."""
    returns = _make_returns(n_months=36, n_assets=4)
    results, _ = run_walk_forward(returns, min_train_months=12)
    assert list(results.columns) == ["Strategy"] + list(returns.columns)
