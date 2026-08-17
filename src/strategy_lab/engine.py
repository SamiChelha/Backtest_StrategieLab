"""
Core Engine Module — strategy_lab.engine
=========================================
Provides multiple portfolio construction strategies and a rigorous
walk-forward backtest engine with full Out-of-Sample (OOS) methodology.

Strategies (all share the same interface)
------------------------------------------
  weights_fn(returns_window: pd.DataFrame,
             min_w: float = 0.0, max_w: float = 1.0,
             **kwargs) -> pd.Series

  Optimisation-based strategies honour min_w / max_w as per-asset
  lower/upper bounds.  Heuristic strategies (Equal Weight, Inverse Vol,
  Momentum, Rank Momentum) accept **kwargs silently for interface
  uniformity but do not apply bounds.

Available strategies
---------------------
  - Equal Weight       (1/N)
  - Minimum Variance     (scipy quadratic optimisation)
  - Maximum Sharpe       (scipy SLSQP)
  - Inverse Volatility   (1/σ proportional)
  - Risk Parity          (equal risk contribution)
  - Momentum (12-1)      (cross-sectional momentum, skip-1-month)
  - Robust Rank Momentum
  - Black-Litterman      (BL posterior + constrained MVO)

Walk-Forward Engine
--------------------
  run_walk_forward(returns, strategy, min_train_months, method,
                   transaction_cost_bps, rebalance_months,
                   min_weight, max_weight)

  method="expanding" — Expanding window (default, rigorous OOS):
    Each month the model is trained on ALL available history up to t-1,
    then tested on month t. The training set grows at every step.
    Every single return in the output is strictly Out-of-Sample.

  method="rolling"   — Rolling window (fixed lookback, legacy):
    The model is trained on the last `min_train_months` only.
    Useful for comparison, but less rigorous.
"""

import itertools

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from strategy_lab.black_litterman import make_bl_strategy
from strategy_lab.metrics import compute_summary


# ---------------------------------------------------------------------------
# Weight-construction helpers (private)
# ---------------------------------------------------------------------------

def _equal_weight(returns_window: pd.DataFrame, **kwargs) -> pd.Series:
    """Equal-weight (1/N) portfolio.  Weight bounds are ignored by design."""
    n = len(returns_window.columns)
    return pd.Series(1.0 / n, index=returns_window.columns)


def _minimum_variance(
    returns_window: pd.DataFrame,
    min_w: float = 0.0,
    max_w: float = 1.0,
    **kwargs,
) -> pd.Series:
    """Minimum variance portfolio via quadratic optimisation.

    Parameters
    ----------
    min_w : float
        Per-asset lower bound (default 0.0 → long-only, unconstrained).
    max_w : float
        Per-asset upper bound (default 1.0 → no concentration cap).
    """
    cols = returns_window.columns
    n = len(cols)
    cov = returns_window.cov().values

    def obj(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(min_w, max_w)] * n
    w0 = np.ones(n) / n
    res = minimize(obj, w0, bounds=bounds, constraints=constraints, method="SLSQP")
    if res.success:
        return pd.Series(res.x, index=cols)
    return pd.Series(w0, index=cols)  # fallback


def _max_sharpe(
    returns_window: pd.DataFrame,
    min_w: float = 0.0,
    max_w: float = 1.0,
    **kwargs,
) -> pd.Series:
    """Maximum Sharpe Ratio portfolio via minimising negative Sharpe.

    Minimises  −(μ · w) / sqrt(w' Σ w)  subject to sum(w)=1,
    min_w ≤ w_i ≤ max_w for all i.

    Risk-free rate is set to 0 (consistent with the rest of the engine).
    Falls back to equal weight on optimisation failure.

    Parameters
    ----------
    min_w : float
        Per-asset lower bound (default 0.0).
    max_w : float
        Per-asset upper bound (default 1.0).
    """
    cols = returns_window.columns
    n = len(cols)
    mean_r = returns_window.mean().values
    cov = returns_window.cov().values

    def neg_sharpe(w):
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 0.0
        return -(w @ mean_r) / port_vol

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_w, max_w)] * n
    res = minimize(
        neg_sharpe, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    weights = res.x if res.success else w0
    return pd.Series(weights, index=cols)


def _inverse_vol(returns_window: pd.DataFrame, **kwargs) -> pd.Series:
    """Inverse-Volatility portfolio.  Weight bounds are ignored by design.

    Each asset receives weight proportional to 1/σ, giving more weight to
    less volatile assets. Assets with zero volatility receive the mean weight.
    """
    vol = returns_window.std()
    safe_vol = vol.replace(0.0, np.nan).fillna(vol[vol > 0].mean()
                                                if (vol > 0).any() else 1.0)
    inv = 1.0 / safe_vol
    return inv / inv.sum()


def _risk_parity(
    returns_window: pd.DataFrame,
    min_w: float = 0.0,
    max_w: float = 1.0,
    **kwargs,
) -> pd.Series:
    """Risk Parity (Equal Risk Contribution) portfolio.

    Each asset contributes equally to total portfolio variance.
    Solved via minimisation of the sum of squared deviations of each
    asset's risk contribution from the target (σ_p / n).

    Falls back to Inverse-Volatility on optimisation failure.

    Parameters
    ----------
    min_w : float
        Per-asset lower bound.  A small positive floor (e.g. 1e-6) is
        always applied internally to avoid division-by-zero in the risk
        contribution formula; the effective lower bound is
        max(min_w, 1e-6).
    max_w : float
        Per-asset upper bound (default 1.0 → no cap).
    """
    cols = returns_window.columns
    n = len(cols)
    cov = returns_window.cov().values

    def risk_contributions(w):
        port_var = w @ cov @ w
        if port_var <= 0:
            return np.zeros(n)
        # Marginal risk contribution: (Σ w)_i * w_i / σ_p
        mrc = cov @ w
        return w * mrc / np.sqrt(port_var)

    def objective(w):
        rc = risk_contributions(w)
        target = rc.sum() / n          # equal share of total risk
        return np.sum((rc - target) ** 2)

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    # Always keep a tiny positive floor to prevent division-by-zero inside
    # the risk-contribution formula, even when the caller passes min_w=0.
    effective_min = max(min_w, 1e-6)
    bounds = [(effective_min, max_w)] * n
    res = minimize(
        objective, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if res.success:
        w = np.maximum(res.x, 0.0)
        return pd.Series(w / w.sum(), index=cols)
    # Fallback: inverse volatility
    return _inverse_vol(returns_window)


def _momentum(returns_window: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Cross-sectional Momentum — "12-1" convention.  Weight bounds ignored.

    Steps:
    1. Use training window returns.
    2. Skip the most recent month (t-1): known short-term reversal effect.
    3. Sum log-returns of each asset over the remaining months → momentum score.
    4. Keep only positive scores (assets that trended up).
    5. Allocate proportionally to positive momentum scores.

    Example (3 assets, 12-month window, skip last):
        SPY: +0.15  →  weight = 0.15 / 0.18 = 83.3%
        AGG: +0.03  →  weight = 0.03 / 0.18 = 16.7%
        GLD: -0.05  →  weight = 0 (negative momentum)
    """
    cols = returns_window.columns
    scores = returns_window.iloc[:-1].sum(axis=0)
    positive = scores.clip(lower=0.0)
    total = positive.sum()
    if total == 0:
        return pd.Series(1.0 / len(cols), index=cols)
    return positive / total


def _rank_momentum(returns_window: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Robust Rank Momentum.  Weight bounds ignored.

    Like Momentum (12-1) but allocates by RANK instead of raw return,
    making the portfolio less sensitive to outlier return values.

    Filters:
    - Top 50% of assets by momentum score only.
    - Assets must also have absolute positive momentum.
    """
    cols = returns_window.columns
    n = len(cols)
    scores = returns_window.iloc[:-1].sum(axis=0)
    ranks = scores.rank(method="first", ascending=True)
    top_half = ranks > (n / 2)
    positive = scores > 0
    mask = top_half & positive
    if not mask.any():
        return pd.Series(1.0 / n, index=cols)
    selected = ranks[mask]
    return selected / selected.sum()


# ---------------------------------------------------------------------------
# Public strategy registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Black-Litterman default instance
# ---------------------------------------------------------------------------
# No explicit views → the strategy uses implied equilibrium returns (π) as
# the sole source of expected returns, giving a well-diversified prior-only
# portfolio.  Replace or extend by calling make_bl_strategy() with views.
#
# Example with views:
#   from strategy_lab.black_litterman import make_bl_strategy
#   STRATEGIES["Black-Litterman"] = make_bl_strategy(
#       views=[
#           {"type": "absolute", "asset": "SPY", "return": 0.07},
#           {"type": "relative",
#            "asset_long": "SPY", "asset_short": "TLT", "return": 0.03},
#       ],
#       w_mkt=None,      # equal-weight prior
#       delta=2.5,
#       tau=0.05,
#       min_weight=0.05,
#       max_weight=0.40,
#   )
_bl_default_inner = make_bl_strategy(
    views=[],          # no views → pure equilibrium prior allocation
    w_mkt=None,
    delta=2.5,
    tau=0.05,
    min_weight=0.05,
    max_weight=0.40,
)

# Thin adapter: the engine passes min_w/max_w at call-time, but BL manages
# its own bounds internally (set at factory time).  We absorb those kwargs
# silently to maintain the uniform strategy interface.
def _bl_default(returns_window: pd.DataFrame, **kwargs) -> pd.Series:
    """Black-Litterman default (no views, equal-weight prior).

    Weight bounds are baked in at factory time (min=0.05, max=0.40).
    The engine's min_w/max_w kwargs are accepted but ignored.
    Use make_bl_strategy() directly to customise bounds or add views.
    """
    return _bl_default_inner(returns_window)

STRATEGIES: dict[str, callable] = {
    "Equal Weight":         _equal_weight,
    "Minimum Variance":     _minimum_variance,
    "Maximum Sharpe":       _max_sharpe,
    "Inverse Volatility":   _inverse_vol,
    "Risk Parity":          _risk_parity,
    "Momentum (12-1)":      _momentum,
    "Robust Rank Momentum": _rank_momentum,
    "Black-Litterman":      _bl_default,
}



# ---------------------------------------------------------------------------
# Walk-forward backtest engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    returns: pd.DataFrame,
    strategy: "str | callable" = "Equal Weight",
    min_train_months: int = 12,
    method: str = "expanding",
    transaction_cost_bps: float = 10.0,
    rebalance_months: int = 1,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    drift_band: float = 0.0,
    rebalancing_type: str = "monthly",
    drift_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest engine — strictly Out-of-Sample.

    Parameters
    ----------
    returns : pd.DataFrame
        Monthly log-return DataFrame (one column per asset).
        The index must be a DatetimeIndex.
    strategy : str or callable
        Name of the strategy — must be a key in STRATEGIES — or a callable
        with the signature ``weights_fn(returns_window, **kwargs) -> pd.Series``.
        String name falls back to Equal Weight if unknown.
    min_train_months : int
        Minimum number of months of history required before the first
        test prediction. The first OOS month is index[min_train_months].
    method : str
        ``"expanding"`` — training window grows at every step (rigorous, default).
        ``"rolling"``   — training window is fixed at ``min_train_months`` (legacy).
    transaction_cost_bps : float
        Round-trip transaction cost in basis points applied to the total
        portfolio turnover at each rebalancing date.
        Example: 10 bps = 0.10% per rebalance.
        Set to 0 to ignore costs.
    rebalance_months : int
        Rebalance every N months (used only when ``rebalancing_type="monthly"``
        with N > 1). For other modes this parameter is ignored.
    min_weight : float
        Per-asset minimum weight enforced at every rebalance.
        Must satisfy ``0.0 <= min_weight < max_weight`` and
        ``min_weight * n_assets <= 1.0`` (feasibility).  Heuristic
        strategies (Equal Weight, Inverse Vol, Momentum, Rank Momentum)
        receive the parameter but ignore it by design.
        Default ``0.0`` preserves the original unconstrained behaviour.
    max_weight : float
        Per-asset maximum weight enforced at every rebalance.
        Must satisfy ``max_weight <= 1.0``.
        Default ``1.0`` preserves the original unconstrained behaviour.
    drift_band : float
        Legacy drift-band threshold (kept for backward compatibility).
        Prefer ``rebalancing_type="drift_band"`` with ``drift_threshold``
        for new code.  Default ``0.0`` disables.
    rebalancing_type : str
        Controls **when** the portfolio actually trades toward target weights w*.
        Target weights are **always recomputed** at every walk-forward step
        regardless of this setting — only the trading decision changes.

        ``"monthly"``    — Rebalance at every walk-forward step (default,
                           preserves existing behaviour).  With
                           ``rebalance_months=N`` you can rebalance every N
                           steps instead.
        ``"daily"``      — Rebalance every step unconditionally (on monthly
                           data this is equivalent to "monthly").
        ``"drift_band"`` — Rebalance only when any asset's live weight deviates
                           more than ``drift_threshold`` from its current target
                           w*.  Reduces unnecessary turnover when asset prices
                           are already close to target.
        ``"quarterly"``  — Rebalance every 3 walk-forward steps (≈ every
                           quarter on monthly data).
    drift_threshold : float
        Deviation threshold used when ``rebalancing_type="drift_band"``.
        An unscheduled rebalance fires when::

            max(|w_current − w_target|) > drift_threshold

        Default 0.05 (5%). Ignored for other rebalancing types.

    Returns
    -------
    results : pd.DataFrame
        Columns: ``'Strategy'`` (net-of-cost OOS log-return) + one weight
        column per asset. Indexed by test date.
    folds_meta : pd.DataFrame
        One row per fold with columns:
        ``'train_start'``, ``'train_end'``, ``'test_date'``,
        ``'rebalanced'`` (bool), ``'turnover'`` (fraction),
        ``'cost_bps'``, ``'drift_triggered'`` (bool),
        ``'rebalancing_type'`` (str).
        Use this to visualise the expanding train/test timeline.

    Raises
    ------
    ValueError
        If the weight bounds are infeasible (see ``min_weight`` description).
    """
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ------------------------------------------------------------------
    # Input validation for weight bounds
    # ------------------------------------------------------------------
    n_assets = len(returns.columns)
    if min_weight < 0.0:
        raise ValueError(
            f"min_weight must be >= 0.0, got {min_weight}."
        )
    if max_weight > 1.0:
        raise ValueError(
            f"max_weight must be <= 1.0, got {max_weight}."
        )
    if min_weight >= max_weight:
        raise ValueError(
            f"min_weight ({min_weight}) must be strictly less than "
            f"max_weight ({max_weight})."
        )
    if min_weight * n_assets > 1.0 + 1e-9:   # small tolerance for floats
        raise ValueError(
            f"Weight constraint is infeasible: min_weight ({min_weight}) × "
            f"n_assets ({n_assets}) = {min_weight * n_assets:.4f} > 1.0. "
            "Lower min_weight or reduce the asset universe."
        )

    _valid_types = {"monthly", "drift_band", "yearly"}
    if rebalancing_type not in _valid_types:
        raise ValueError(
            f"rebalancing_type must be one of {_valid_types}, "
            f"got '{rebalancing_type}'."
        )

    if callable(strategy):
        weight_fn = strategy
    else:
        weight_fn = STRATEGIES.get(strategy, _equal_weight)
    cols = returns.columns
    n = len(cols)

    strat_rets: list[float] = []
    weights_hist: list[pd.Series] = []
    dates: list = []
    folds: list[dict] = []

    # Current (live, drifted) portfolio weights — initialised to equal weight
    w = pd.Series(1.0 / n, index=cols)
    # Track last rebalanced weights (for legacy drift_band checks)
    w_last_rebalanced = w.copy()
    # Step counter within the OOS period (for quarterly gating)
    oos_step = 0

    for i in range(len(returns)):
        if i < min_train_months:
            continue  # not enough training history yet

        # ------------------------------------------------------------------
        # Build training window
        # ------------------------------------------------------------------
        if method == "rolling":
            window = returns.iloc[i - min_train_months: i]
        else:  # expanding (default)
            window = returns.iloc[0: i]

        train_start = window.index[0]
        train_end   = window.index[-1]
        test_date   = returns.index[i]

        # ------------------------------------------------------------------
        # ALWAYS recompute target weights w* at every step.
        # rebalancing_type only controls whether the portfolio TRADES.
        # ------------------------------------------------------------------
        w_target = weight_fn(window, min_w=min_weight, max_w=max_weight)

        # ------------------------------------------------------------------
        # Determine whether to rebalance this step
        # ------------------------------------------------------------------
        drift_triggered = False

        if rebalancing_type == "monthly":
            # Rebalance every N steps (existing behaviour)
            do_rebalance = (oos_step % rebalance_months == 0)

        elif rebalancing_type == "yearly":
            # Rebalance every 12 OOS steps (≈ annually on monthly data)
            do_rebalance = (oos_step % 12 == 0)

        elif rebalancing_type == "drift_band":
            # Rebalance when any asset deviates > drift_threshold from w*
            max_drift = float((w - w_target).abs().max())
            do_rebalance = (max_drift > drift_threshold)
            if do_rebalance:
                drift_triggered = True

        else:
            do_rebalance = True  # unreachable after validation above

        # Legacy drift_band parameter (backward compat): fire an extra
        # unscheduled rebalance if drifted from last-rebalanced weights.
        if not do_rebalance and drift_band > 0.0:
            max_drift_legacy = float((w - w_last_rebalanced).abs().max())
            if max_drift_legacy > drift_band:
                do_rebalance = True
                drift_triggered = True

        # ------------------------------------------------------------------
        # Execute trade (or not)
        # ------------------------------------------------------------------
        if do_rebalance:
            turnover = float((w_target - w).abs().sum()) / 2.0
            cost = turnover * (transaction_cost_bps / 10_000)
            w = w_target.copy()
            w_last_rebalanced = w.copy()
        else:
            turnover = 0.0
            cost = 0.0

        # ------------------------------------------------------------------
        # OOS return for this month
        # ------------------------------------------------------------------
        row = returns.iloc[i]
        gross_ret = float((row * w).sum())
        net_ret = gross_ret - cost

        strat_rets.append(net_ret)
        weights_hist.append(w.copy())
        dates.append(test_date)

        folds.append({
            "train_start":      train_start,
            "train_end":        train_end,
            "test_date":        test_date,
            "rebalanced":       do_rebalance,
            "turnover":         round(turnover, 4),
            "cost_bps":         round(cost * 10_000, 2),
            "drift_triggered":  drift_triggered,
            "rebalancing_type": rebalancing_type,
        })

        # ------------------------------------------------------------------
        # Mark-to-market drift: w_{t+1} ∝ w_t * exp(r_t)
        # ------------------------------------------------------------------
        drifted = w * np.exp(row)
        s = drifted.sum()
        if s > 0:
            w = drifted / s

        oos_step += 1

    if not strat_rets:
        return pd.DataFrame(), pd.DataFrame()

    results = pd.DataFrame(weights_hist, index=dates, columns=cols)
    results.insert(0, "Strategy", pd.Series(strat_rets, index=dates))

    folds_meta = pd.DataFrame(folds)

    return results, folds_meta


# ---------------------------------------------------------------------------
# Impact-of-constraints analysis utility
# ---------------------------------------------------------------------------

def analyze_weight_constraints(
    returns: pd.DataFrame,
    strategy: str,
    min_weights: list[float],
    max_weight: float = 1.0,
    min_train_months: int = 12,
    transaction_cost_bps: float = 10.0,
    rebalance_months: int = 1,
    method: str = "expanding",
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Analyse the impact of minimum-weight constraints on portfolio metrics.

    Runs the walk-forward backtest for each value in ``min_weights`` and
    collects a set of risk/return/concentration statistics, returning them
    as a tidy DataFrame.  Directly supports Step 3 of the academic project:
    *analyse the effect of weight constraints on risk, return and
    concentration*.

    Parameters
    ----------
    returns : pd.DataFrame
        Monthly log-return DataFrame (one column per asset).
    strategy : str
        Strategy name — must be a key in ``STRATEGIES``.
    min_weights : list[float]
        Sequence of per-asset lower bounds to test,
        e.g. ``[0.0, 0.05, 0.10, 0.20]``.
        Values that fail the feasibility check
        (``min_w * n_assets > 1``) are skipped with a warning.
    max_weight : float
        Per-asset upper bound applied in every run.  Default 1.0 (no cap).
    min_train_months : int
        Passed through to ``run_walk_forward``.  Default 12.
    transaction_cost_bps : float
        Transaction cost in basis points.  Default 10.
    rebalance_months : int
        Rebalancing frequency.  Default 1 (monthly).
    method : str
        ``"expanding"`` (default) or ``"rolling"``.
    periods_per_year : int
        Used to annualise volatility.  Default 12 (monthly data).

    Returns
    -------
    pd.DataFrame
        One row per valid ``min_weight`` value.  Columns:

        ``min_weight``
            The lower bound tested.
        ``annualized_return``
            CAGR: ``exp(mean(r) * periods_per_year) − 1``.
        ``volatility``
            Annualised portfolio standard deviation.
        ``sharpe``
            Annualised Sharpe ratio (rf = 0).
        ``max_drawdown``
            Peak-to-trough drawdown (negative fraction).
        ``avg_concentration``
            Mean of the *maximum single-asset weight* across all
            rebalancing dates — a simple HHI proxy for concentration.
            Higher values indicate more concentrated portfolios.

    Notes
    -----
    Infeasible ``min_weight`` values (i.e. ``min_w * n_assets > 1``) are
    silently skipped and excluded from the output DataFrame.

    Examples
    --------
    >>> df = analyze_weight_constraints(
    ...     returns=monthly_returns,
    ...     strategy="Minimum Variance",
    ...     min_weights=[0.0, 0.05, 0.10, 0.20],
    ...     max_weight=1.0,
    ... )
    >>> print(df.to_string(index=False))
    """
    n_assets = len(returns.columns)
    records: list[dict] = []

    for min_w in min_weights:
        # Skip infeasible bounds silently
        if min_w * n_assets > 1.0 + 1e-9:
            continue
        if min_w < 0.0 or min_w >= max_weight:
            continue

        try:
            results, _ = run_walk_forward(
                returns=returns,
                strategy=strategy,
                min_train_months=min_train_months,
                method=method,
                transaction_cost_bps=transaction_cost_bps,
                rebalance_months=rebalance_months,
                min_weight=min_w,
                max_weight=max_weight,
            )
        except (ValueError, Exception):
            continue

        if results.empty:
            continue

        rets = results["Strategy"]
        ann_ret  = float(np.exp(rets.mean() * periods_per_year) - 1)
        vol      = float(rets.std() * np.sqrt(periods_per_year))
        sharpe   = float(rets.mean() / rets.std() * np.sqrt(periods_per_year))\
                   if rets.std() > 0 else float("nan")

        # Max drawdown
        wealth = np.exp(rets.cumsum())
        peaks  = wealth.cummax()
        mdd    = float(((wealth - peaks) / peaks).min())

        # Concentration: mean of the per-date maximum weight
        weight_cols = [c for c in results.columns if c != "Strategy"]
        avg_conc = float(results[weight_cols].max(axis=1).mean())

        records.append({
            "min_weight":        round(min_w, 6),
            "annualized_return": round(ann_ret, 6),
            "volatility":        round(vol, 6),
            "sharpe":            round(sharpe, 4),
            "max_drawdown":      round(mdd, 6),
            "avg_concentration": round(avg_conc, 4),
        })

    return pd.DataFrame(
        records,
        columns=[
            "min_weight",
            "annualized_return",
            "volatility",
            "sharpe",
            "max_drawdown",
            "avg_concentration",
        ],
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis utility
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    returns: pd.DataFrame,
    views: list[dict],
    w_mkt: "pd.Series | None" = None,
    delta_values: list[float] = [1.5, 2.0, 2.5, 3.0, 4.0],
    tau_values: list[float]   = [0.01, 0.05, 0.10, 0.20],
    view_confidences: "list[float] | None" = None,
    min_weight: float = 0.05,
    max_weight: float = 0.45,
    min_train_months: int = 12,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Sensitivity analysis over Black-Litterman hyper-parameters δ and τ.

    Runs the full walk-forward backtest for every (delta, tau) combination
    in the Cartesian product of ``delta_values`` × ``tau_values`` and
    collects key performance metrics.  Useful for understanding how robust
    BL portfolio performance is to the choice of risk-aversion and prior
    uncertainty parameters.

    Parameters
    ----------
    returns : pd.DataFrame
        Monthly log-return DataFrame (one column per asset).
    views : list[dict]
        Black-Litterman view dictionaries passed to ``make_bl_strategy``.
    w_mkt : pd.Series or None
        Market-cap weights for the BL prior.  Pass ``None`` for equal-weight.
    delta_values : list[float]
        Risk-aversion coefficients to sweep.
        Default: ``[1.5, 2.0, 2.5, 3.0, 4.0]``.
    tau_values : list[float]
        Prior-uncertainty scalars to sweep.
        Default: ``[0.01, 0.05, 0.10, 0.20]``.
    view_confidences : list[float] or None
        Reserved for future use (currently unused).  Pass ``None``.
    min_weight : float
        Per-asset minimum weight passed to ``make_bl_strategy``.
        Default ``0.05``.
    max_weight : float
        Per-asset maximum weight passed to ``make_bl_strategy``.
        Default ``0.45``.
    min_train_months : int
        Minimum training months for ``run_walk_forward``.  Default 12.
    transaction_cost_bps : float
        Transaction cost in basis points.  Default 10.

    Returns
    -------
    pd.DataFrame
        MultiIndex (delta, tau) with metric columns:
        ``annualized_return``, ``sharpe``, ``sortino``,
        ``max_drawdown``, ``calmar``, ``total_return``.
        Combinations that raise an exception are silently skipped.

    Examples
    --------
    >>> sens = run_sensitivity_analysis(
    ...     returns=monthly_returns,
    ...     views=[{"type": "absolute", "asset": "SPY", "return": 0.07}],
    ... )
    >>> print(sens)
    """
    records = []

    for delta, tau in itertools.product(delta_values, tau_values):
        try:
            bl_strat = make_bl_strategy(
                views=views,
                w_mkt=w_mkt,
                delta=delta,
                tau=tau,
                min_weight=min_weight,
                max_weight=max_weight,
            )
            results, _ = run_walk_forward(
                returns=returns,
                strategy=bl_strat,
                min_train_months=min_train_months,
                transaction_cost_bps=transaction_cost_bps,
                min_weight=min_weight,
                max_weight=max_weight,
            )
            if results.empty:
                continue
            summary = compute_summary(results["Strategy"])
            records.append({
                "delta":             delta,
                "tau":               tau,
                "annualized_return": summary["annualized_return"],
                "sharpe":            summary["sharpe"],
                "sortino":           summary["sortino"],
                "max_drawdown":      summary["max_drawdown"],
                "calmar":            summary["calmar"],
                "total_return":      summary["total_return"],
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame(
            columns=["delta", "tau", "annualized_return", "sharpe",
                     "sortino", "max_drawdown", "calmar", "total_return"]
        ).set_index(["delta", "tau"])

    df = pd.DataFrame(records)
    df = df.set_index(["delta", "tau"])
    return df
