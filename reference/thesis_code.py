"""
Thesis code — Performance metrics (CVaR/ES, Sharpe, Sortino, Calmar, Beta, IR)
             + Walk-Forward backtest (expanding & rolling windows).
Dependencies: numpy, pandas, scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =============================================================================
# 1. CVaR / Expected Shortfall (ES)
# =============================================================================

def cvar(log_returns: pd.Series, alpha: float = 0.05, periods_per_year: int = 12) -> float:
    """Average return in the worst alpha-fraction of months (annualised).
    alpha=0.05 → 95% CVaR (expected loss beyond the 5% worst cases).
    """
    if len(log_returns) < 2 or alpha <= 0:
        return float("nan")
    tail = log_returns[log_returns <= log_returns.quantile(alpha)]
    if tail.empty:
        return float("nan")
    return float(tail.mean()) * np.sqrt(periods_per_year)


def var(log_returns: pd.Series, alpha: float = 0.05, periods_per_year: int = 12) -> float:
    """Alpha-quantile of the return distribution (annualised). Companion to CVaR."""
    if len(log_returns) < 2:
        return float("nan")
    return float(log_returns.quantile(alpha)) * np.sqrt(periods_per_year)


def cvar_summary(
    log_returns: pd.Series,
    alphas: list[float] | None = None,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """VaR and CVaR at 90%, 95%, 99% confidence (or custom alphas). All annualised."""
    if alphas is None:
        alphas = [0.10, 0.05, 0.01]
    return pd.DataFrame([{
        "alpha":      a,
        "confidence": f"{(1 - a):.0%}",
        "VaR":        var(log_returns, a, periods_per_year),
        "CVaR (ES)":  cvar(log_returns, a, periods_per_year),
    } for a in alphas])


# =============================================================================
# 2. Performance metrics
# =============================================================================

def annualized_return(log_returns: pd.Series, periods_per_year: int = 12) -> float:
    """CAGR: exp(mean(r) * N) - 1. Returns e.g. 0.08 for 8% per year."""
    if len(log_returns) == 0:
        return float("nan")
    return float(np.exp(log_returns.mean() * periods_per_year) - 1)


def sharpe_ratio(
    log_returns: pd.Series,
    rf: pd.Series | None = None,
    periods_per_year: int = 12,
) -> float:
    """Annualised return per unit of total volatility. rf=None means risk-free = 0."""
    if len(log_returns) == 0:
        return float("nan")
    excess = log_returns - rf.reindex(log_returns.index).fillna(0.0) if rf is not None else log_returns
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std()) if excess.std() > 0 else float("nan")


def sortino_ratio(
    log_returns: pd.Series,
    rf: pd.Series | None = None,
    periods_per_year: int = 12,
) -> float:
    """Like Sharpe but divides by downside volatility only — doesn't penalise upside gains."""
    if len(log_returns) == 0:
        return float("nan")
    excess = log_returns - rf.reindex(log_returns.index).fillna(0.0) if rf is not None else log_returns
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std())


def max_drawdown(log_returns: pd.Series) -> float:
    """Largest peak-to-trough loss as a negative fraction (e.g. -0.35 = -35%)."""
    if len(log_returns) == 0:
        return float("nan")
    wealth = np.exp(log_returns.cumsum())
    return float(((wealth - wealth.cummax()) / wealth.cummax()).min())


def calmar_ratio(log_returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised return divided by max drawdown — return per unit of worst loss."""
    ann = annualized_return(log_returns, periods_per_year)
    mdd = max_drawdown(log_returns)
    if np.isnan(mdd) or mdd == 0:
        return float("nan")
    return float(ann / abs(mdd))


def beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Market sensitivity vs a benchmark: cov(s, b) / var(b).
    1.0 = moves with market, >1 = amplifies, <1 = defensive, <0 = hedge.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    var_bench = aligned.iloc[:, 1].var()
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / var_bench) if var_bench > 0 else float("nan")


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """Active return (alpha) per unit of tracking error vs a benchmark.
    IR > 0.5 = good active management, IR > 1.0 = excellent.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(np.sqrt(periods_per_year) * active.mean() / active.std()) if active.std() > 0 else float("nan")


def compute_summary(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    rf: pd.Series | None = None,
    periods_per_year: int = 12,
) -> dict:
    """All metrics in one call. benchmark and rf are optional (NaN if not provided)."""
    wealth = np.exp(strategy_returns.cumsum())
    return {
        "annualized_return": annualized_return(strategy_returns, periods_per_year),
        "sharpe":            sharpe_ratio(strategy_returns, rf, periods_per_year),
        "sortino":           sortino_ratio(strategy_returns, rf, periods_per_year),
        "max_drawdown":      max_drawdown(strategy_returns),
        "calmar":            calmar_ratio(strategy_returns, periods_per_year),
        "beta":              beta(strategy_returns, benchmark_returns) if benchmark_returns is not None else float("nan"),
        "information_ratio": information_ratio(strategy_returns, benchmark_returns, periods_per_year) if benchmark_returns is not None else float("nan"),
        "CVaR (ES) 95%":     cvar(strategy_returns, alpha=0.05, periods_per_year=periods_per_year),
        "total_return":      float(wealth.iloc[-1] - 1.0) if len(wealth) > 0 else float("nan"),
        "n_months":          len(strategy_returns),
    }


# =============================================================================
# 4. Portfolio strategies
# =============================================================================

def _equal_weight(returns_window: pd.DataFrame, **kwargs) -> pd.Series:
    """1/N across all assets."""
    n = len(returns_window.columns)
    return pd.Series(1.0 / n, index=returns_window.columns)


def _minimum_variance(
    returns_window: pd.DataFrame, min_w: float = 0.0, max_w: float = 1.0, **kwargs
) -> pd.Series:
    """Minimise portfolio variance subject to sum(w)=1 and per-asset bounds."""
    cols = returns_window.columns
    n = len(cols)
    cov = returns_window.cov().values
    w0 = np.ones(n) / n
    res = minimize(
        lambda w: w @ cov @ w, w0,
        bounds=[(min_w, max_w)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        method="SLSQP",
    )
    return pd.Series(res.x if res.success else w0, index=cols)


def _max_sharpe(
    returns_window: pd.DataFrame, min_w: float = 0.0, max_w: float = 1.0, **kwargs
) -> pd.Series:
    """Maximise Sharpe ratio (rf = 0)."""
    cols = returns_window.columns
    n = len(cols)
    mean_r = returns_window.mean().values
    cov = returns_window.cov().values

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return 0.0 if vol < 1e-10 else -(w @ mean_r) / vol

    w0 = np.ones(n) / n
    res = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=[(min_w, max_w)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    return pd.Series(res.x if res.success else w0, index=cols)


def _inverse_vol(returns_window: pd.DataFrame, **kwargs) -> pd.Series:
    """Weight proportional to 1/volatility — less volatile assets get more weight."""
    vol = returns_window.std()
    safe = vol.replace(0.0, np.nan).fillna(vol[vol > 0].mean() if (vol > 0).any() else 1.0)
    inv = 1.0 / safe
    return inv / inv.sum()


def _risk_parity(
    returns_window: pd.DataFrame, min_w: float = 0.0, max_w: float = 1.0, **kwargs
) -> pd.Series:
    """Each asset contributes equally to total portfolio risk."""
    cols = returns_window.columns
    n = len(cols)
    cov = returns_window.cov().values

    def objective(w):
        pv = w @ cov @ w
        if pv <= 0:
            return 0.0
        rc = w * (cov @ w) / np.sqrt(pv)
        return np.sum((rc - rc.sum() / n) ** 2)

    w0 = np.ones(n) / n
    res = minimize(
        objective, w0, method="SLSQP",
        bounds=[(max(min_w, 1e-6), max_w)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if res.success:
        w = np.maximum(res.x, 0.0)
        return pd.Series(w / w.sum(), index=cols)
    return _inverse_vol(returns_window)


STRATEGIES: dict[str, callable] = {
    "Equal Weight":       _equal_weight,
    "Minimum Variance":   _minimum_variance,
    "Maximum Sharpe":     _max_sharpe,
    "Inverse Volatility": _inverse_vol,
    "Risk Parity":        _risk_parity,
}


# =============================================================================
# 5. Walk-Forward Engine
# =============================================================================
#
#   expanding  Train on ALL history up to t-1. Window grows each step.
#              Rigorous OOS — every prediction uses only past data.
#
#              [Jan Feb Mar] → test Apr
#              [Jan Feb Mar Apr] → test May   (start stays fixed)
#
#   rolling    Train on the last N months only. Window slides forward.
#              Useful for comparison; ignores older data.
#
#              [Jan Feb Mar] → test Apr
#              [Feb Mar Apr] → test May       (start slides forward)
# =============================================================================

def run_expanding_window(
    returns: pd.DataFrame,
    strategy: "str | callable" = "Equal Weight",
    min_train_months: int = 12,
    transaction_cost_bps: float = 10.0,
    rebalance_months: int = 1,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward backtest — expanding window (growing training set)."""
    return _walk_forward(
        returns, strategy, min_train_months, "expanding",
        transaction_cost_bps, rebalance_months, min_weight, max_weight,
    )


def run_rolling_window(
    returns: pd.DataFrame,
    strategy: "str | callable" = "Equal Weight",
    min_train_months: int = 12,
    transaction_cost_bps: float = 10.0,
    rebalance_months: int = 1,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward backtest — rolling window (fixed-length training set)."""
    return _walk_forward(
        returns, strategy, min_train_months, "rolling",
        transaction_cost_bps, rebalance_months, min_weight, max_weight,
    )


def _walk_forward(
    returns: pd.DataFrame,
    strategy: "str | callable",
    min_train_months: int,
    method: str,           # "expanding" or "rolling"
    transaction_cost_bps: float,
    rebalance_months: int,
    min_weight: float,
    max_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared walk-forward loop for both window methods.

    At each step:
      1. Build training window (expanding or rolling).
      2. Compute target weights w* from that window.
      3. Rebalance if scheduled; apply transaction cost on turnover.
      4. Record OOS return (net of cost) and let weights drift with prices.
    """
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    n_assets = len(returns.columns)
    if min_weight * n_assets > 1.0 + 1e-9:
        raise ValueError(
            f"Infeasible: min_weight ({min_weight}) × n_assets ({n_assets}) > 1.0"
        )

    weight_fn = strategy if callable(strategy) else STRATEGIES.get(strategy, _equal_weight)
    cols = returns.columns
    n = len(cols)

    strat_rets: list[float] = []
    weights_hist: list[pd.Series] = []
    dates: list = []
    folds: list[dict] = []

    w = pd.Series(1.0 / n, index=cols)  # current live weights (drift between rebalances)
    oos_step = 0

    for i in range(len(returns)):
        if i < min_train_months:
            continue

        # --- 1. Build training window ---
        if method == "rolling":
            window = returns.iloc[i - min_train_months: i]  # fixed last N months
        else:
            window = returns.iloc[0: i]                     # all history so far

        train_start = window.index[0]
        train_end   = window.index[-1]
        test_date   = returns.index[i]

        # --- 2. Target weights from training window ---
        w_target = weight_fn(window, min_w=min_weight, max_w=max_weight)

        # --- 3. Rebalance if scheduled ---
        do_rebalance = (oos_step % rebalance_months == 0)
        if do_rebalance:
            turnover = float((w_target - w).abs().sum()) / 2.0
            cost = turnover * (transaction_cost_bps / 10_000)
            w = w_target.copy()
        else:
            turnover = 0.0
            cost = 0.0

        # --- 4. OOS return (net of cost) ---
        row = returns.iloc[i]
        net_ret = float((row * w).sum()) - cost

        strat_rets.append(net_ret)
        weights_hist.append(w.copy())
        dates.append(test_date)
        folds.append({
            "train_start": train_start,
            "train_end":   train_end,
            "test_date":   test_date,
            "method":      method,
            "rebalanced":  do_rebalance,
            "turnover":    round(turnover, 4),
            "cost_bps":    round(cost * 10_000, 2),
        })

        # Natural price drift between rebalances: w ∝ w * exp(r)
        drifted = w * np.exp(row)
        s = drifted.sum()
        if s > 0:
            w = drifted / s

        oos_step += 1

    if not strat_rets:
        return pd.DataFrame(), pd.DataFrame()

    results = pd.DataFrame(weights_hist, index=dates, columns=cols)
    results.insert(0, "Strategy", pd.Series(strat_rets, index=dates))
    return results, pd.DataFrame(folds)


# =============================================================================
# 6. Compare expanding vs rolling (full metrics table)
# =============================================================================

def compare_expanding_vs_rolling(
    returns: pd.DataFrame,
    strategy: "str | callable" = "Equal Weight",
    min_train_months: int = 12,
    transaction_cost_bps: float = 10.0,
    rebalance_months: int = 1,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    benchmark_returns: pd.Series | None = None,
    rf: pd.Series | None = None,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Run both window methods and return a full side-by-side metrics table."""
    records = {}
    for method, run_fn in [("expanding", run_expanding_window), ("rolling", run_rolling_window)]:
        res, _ = run_fn(
            returns, strategy, min_train_months,
            transaction_cost_bps, rebalance_months, min_weight, max_weight,
        )
        if res.empty:
            continue
        summary = compute_summary(res["Strategy"], benchmark_returns, rf, periods_per_year)
        records[method] = {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}
    return pd.DataFrame(records).T


# =============================================================================
# 7. Quick demo (python thesis_code.py)
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    assets = ["SPY", "TLT", "GLD", "AGG"]
    dates  = pd.date_range("2015-01-31", periods=120, freq="ME")
    returns = pd.DataFrame(
        rng.multivariate_normal(
            [0.008, 0.003, 0.004, 0.002],
            [[ 0.0020, -0.0006,  0.0002, -0.0003],
             [-0.0006,  0.0012,  0.0001,  0.0004],
             [ 0.0002,  0.0001,  0.0018,  0.0001],
             [-0.0003,  0.0004,  0.0001,  0.0006]],
            120,
        ),
        index=dates, columns=assets,
    )

    # --- CVaR / ES table ---
    print("=" * 60)
    print("CVaR / ES — Equal Weight (buy-and-hold)")
    print("=" * 60)
    print(cvar_summary(returns.mean(axis=1)).to_string(index=False))
    print()

    # --- All metrics for one strategy ---
    res_exp, _ = run_expanding_window(returns, "Minimum Variance", min_train_months=24)
    print("=" * 60)
    print("Full metrics — Minimum Variance (expanding window)")
    print("=" * 60)
    for k, v in compute_summary(res_exp["Strategy"]).items():
        print(f"  {k:<22} {v:.4f}" if isinstance(v, float) else f"  {k:<22} {v}")
    print()

    # --- Expanding vs Rolling comparison ---
    print("=" * 60)
    print("Expanding vs Rolling — Minimum Variance")
    print("=" * 60)
    print(compare_expanding_vs_rolling(returns, "Minimum Variance", min_train_months=24).to_string())
    print()

    # --- Fold metadata ---
    print("=" * 60)
    print("Fold metadata — Expanding (first 5)")
    print("=" * 60)
    _, folds_e = run_expanding_window(returns, "Minimum Variance", min_train_months=24)
    print(folds_e[["train_start", "train_end", "test_date", "method", "turnover"]].head().to_string(index=False))
    print()

    print("=" * 60)
    print("Fold metadata — Rolling (first 5)")
    print("=" * 60)
    _, folds_r = run_rolling_window(returns, "Minimum Variance", min_train_months=24)
    print(folds_r[["train_start", "train_end", "test_date", "method", "turnover"]].head().to_string(index=False))
