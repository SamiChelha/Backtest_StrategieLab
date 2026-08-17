"""
Black-Litterman Module — strategy_lab.black_litterman
======================================================
Implements the full Black-Litterman (1990) framework for portfolio construction.

Pipeline
--------
  1. Implied equilibrium returns  π = δ Σ w_mkt           (reverse optimisation)
  2. View matrices P, Q, Ω        (absolute + relative views)
  3. BL posterior returns & covariance
  4. Constrained mean-variance optimisation with BL posterior
  5. Strategy factory compatible with the engine's STRATEGIES registry
  6. Comparison utility vs. classic Markowitz

Public API
----------
  make_bl_strategy(views, w_mkt, delta, tau, min_weight, max_weight)
      → Callable[[pd.DataFrame], pd.Series]

  compare_bl_vs_markowitz(returns_window, views, w_mkt)
      → pd.DataFrame

References
----------
  Black, F. & Litterman, R. (1992). "Global Portfolio Optimization."
  Financial Analysts Journal, 48(5), 28-43.

  He, G. & Litterman, R. (1999). "The Intuition Behind Black-Litterman
  Model Portfolios." Goldman Sachs Investment Management.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ViewDict = dict  # {"type": "absolute"|"relative", ...}


# ---------------------------------------------------------------------------
# Step 1 — Implied Equilibrium Returns (Reverse Optimisation)
# ---------------------------------------------------------------------------

def implied_equilibrium_returns(
    cov: np.ndarray,
    w_mkt: np.ndarray,
    delta: float = 2.5,
) -> np.ndarray:
    """Compute implied equilibrium excess returns via reverse optimisation.

    Solves  π = δ · Σ · w_mkt, which is the first-order condition of a
    mean-variance investor holding the market portfolio.

    Parameters
    ----------
    cov : np.ndarray, shape (N, N)
        Asset covariance matrix (annualised or at the same frequency as
        the returns window — must be consistent throughout).
    w_mkt : np.ndarray, shape (N,)
        Market-capitalisation weights (must sum to 1).
    delta : float
        Risk-aversion coefficient.  Typical range 1–4, default 2.5.

    Returns
    -------
    pi : np.ndarray, shape (N,)
        Implied equilibrium excess returns (same units as the covariance).

    Notes
    -----
    Weights are L1-normalised before computation to guarantee they sum to 1
    even if the caller passes unnormalised values.
    """
    w_safe = np.asarray(w_mkt, dtype=float)
    w_safe = w_safe / w_safe.sum()          # normalise
    pi: np.ndarray = delta * cov @ w_safe
    return pi


# ---------------------------------------------------------------------------
# Step 2 — View Matrices (P, Q, Ω)
# ---------------------------------------------------------------------------

def build_view_matrices(
    views: list[ViewDict],
    assets: list[str],
    cov: np.ndarray,
    tau: float = 0.05,
    confidences: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the P, Q, and Ω matrices from a list of view dictionaries.

    Supported view types
    --------------------
    Absolute view::

        {"type": "absolute", "asset": "SPY", "return": 0.07}
        → P row: [1, 0, 0, 0], Q entry: 0.07

    Relative view::

        {"type": "relative", "asset_long": "SPY",
         "asset_short": "TLT", "return": 0.03}
        → P row: [1, 0, -1, 0], Q entry: 0.03

    The He-Litterman proportional uncertainty method is used to set Ω::

        Ω_ii = (1/c_i − 1) · τ · (P_i Σ P_iᵀ)

    where c_i is the per-view confidence (default 0.5).  At c=0.5 this
    reduces to the standard formula Ω = diag(τ · P Σ Pᵀ).  At c→1 the
    uncertainty vanishes (BL posterior → view exactly); at c→0 the
    uncertainty is very large (BL posterior → equilibrium prior).

    Parameters
    ----------
    views : list[dict]
        List of view dictionaries (see formats above).
    assets : list[str]
        Ordered list of asset names corresponding to the covariance columns.
    cov : np.ndarray, shape (N, N)
        Asset covariance matrix used to scale view uncertainties.
    tau : float
        Scalar expressing confidence in the prior relative to the views.
        Smaller τ → more weight on the prior (equilibrium).  Default 0.05.
    confidences : list[float] or None
        Per-view confidence levels in [0, 1].  ``None`` defaults to 0.5 for
        every view (standard He-Litterman proportional uncertainty).

    Returns
    -------
    P : np.ndarray, shape (K, N)
        View-picking matrix.  K = number of views, N = number of assets.
    Q : np.ndarray, shape (K,)
        Expected return for each view.
    Omega : np.ndarray, shape (K, K)
        Diagonal view uncertainty matrix.

    Raises
    ------
    ValueError
        If a view references an asset not present in ``assets``, or if
        ``views`` is empty.
    """
    if not views:
        raise ValueError(
            "At least one view must be provided to build the view matrices."
        )

    asset_index: dict[str, int] = {a: i for i, a in enumerate(assets)}
    n = len(assets)
    k = len(views)

    P = np.zeros((k, n), dtype=float)
    Q = np.zeros(k, dtype=float)

    for row_idx, view in enumerate(views):
        vtype = view.get("type", "").lower()

        if vtype == "absolute":
            asset = view["asset"]
            if asset not in asset_index:
                raise ValueError(
                    f"Absolute view references unknown asset '{asset}'. "
                    f"Available: {assets}"
                )
            P[row_idx, asset_index[asset]] = 1.0
            Q[row_idx] = float(view["return"])

        elif vtype == "relative":
            long_asset  = view["asset_long"]
            short_asset = view["asset_short"]
            for a in (long_asset, short_asset):
                if a not in asset_index:
                    raise ValueError(
                        f"Relative view references unknown asset '{a}'. "
                        f"Available: {assets}"
                    )
            P[row_idx, asset_index[long_asset]]  =  1.0
            P[row_idx, asset_index[short_asset]] = -1.0
            Q[row_idx] = float(view["return"])

        else:
            raise ValueError(
                f"Unknown view type '{vtype}'. "
                "Supported: 'absolute', 'relative'."
            )

    # Per-view confidence levels — default to 0.5 (standard He-Litterman)
    if confidences is None:
        c_arr = np.full(k, 0.5)
    else:
        c_arr = np.clip(np.asarray(confidences, dtype=float), 1e-6, 1.0 - 1e-6)

    # He-Litterman proportional uncertainty scaled by confidence:
    #   Ω_ii = (1/c_i − 1) · τ · (P_i Σ P_iᵀ)
    # At c=0.5: scale = 1.0  → identical to the unscaled formula
    # At c→1:   scale → 0   → views dominate (low uncertainty)
    # At c→0:   scale → ∞   → prior dominates (high uncertainty)
    base_diag = tau * np.diag(P @ cov @ P.T)
    scales = (1.0 / c_arr) - 1.0
    omega_diag = scales * base_diag
    Omega = np.diag(omega_diag)

    return P, Q, Omega


# ---------------------------------------------------------------------------
# Step 3 — Black-Litterman Posterior
# ---------------------------------------------------------------------------

def bl_posterior(
    pi: np.ndarray,
    cov: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Black-Litterman posterior expected returns and covariance.

    Using the analytic form of Bayes' theorem for Gaussian priors and
    Gaussian view likelihoods:

    .. math::

        \\mu_{BL} = \\left[(\\tau\\Sigma)^{-1} + P^\\top\\Omega^{-1}P\\right]^{-1}
                   \\left[(\\tau\\Sigma)^{-1}\\pi + P^\\top\\Omega^{-1}Q\\right]

        M^{-1} = \\left[(\\tau\\Sigma)^{-1} + P^\\top\\Omega^{-1}P\\right]^{-1}

        \\Sigma_{BL} = \\Sigma + M^{-1}

    Parameters
    ----------
    pi : np.ndarray, shape (N,)
        Implied equilibrium excess returns (prior mean).
    cov : np.ndarray, shape (N, N)
        Asset covariance matrix.
    P : np.ndarray, shape (K, N)
        View-picking matrix.
    Q : np.ndarray, shape (K,)
        View expected returns.
    Omega : np.ndarray, shape (K, K)
        View uncertainty (diagonal) matrix.
    tau : float
        Scaling parameter for the prior uncertainty.

    Returns
    -------
    mu_bl : np.ndarray, shape (N,)
        BL posterior expected returns.
    cov_bl : np.ndarray, shape (N, N)
        BL posterior covariance matrix.

    Notes
    -----
    Uses ``np.linalg.solve`` in preference to explicit matrix inversion for
    numerical stability.
    """
    tau_cov      = tau * cov
    tau_cov_inv  = np.linalg.inv(tau_cov)
    omega_inv    = np.linalg.inv(Omega)

    # Posterior precision matrix  M = (τΣ)^{-1} + Pᵀ Ω^{-1} P
    M = tau_cov_inv + P.T @ omega_inv @ P

    # Posterior mean  μ_BL = M^{-1} [(τΣ)^{-1} π + Pᵀ Ω^{-1} Q]
    rhs    = tau_cov_inv @ pi + P.T @ omega_inv @ Q
    mu_bl  = np.linalg.solve(M, rhs)

    # Posterior covariance  Σ_BL = Σ + M^{-1}
    M_inv  = np.linalg.inv(M)
    cov_bl = cov + M_inv

    return mu_bl, cov_bl


# ---------------------------------------------------------------------------
# Step 4 — Constrained Mean-Variance Optimisation
# ---------------------------------------------------------------------------

def bl_optimal_weights(
    mu_bl: np.ndarray,
    cov_bl: np.ndarray,
    assets: list[str],
    delta: float = 2.5,
    min_weight: float = 0.05,
    max_weight: float = 0.40,
) -> pd.Series:
    """Maximise the Black-Litterman mean-variance utility via SLSQP.

    Objective (to minimise):

    .. math::

        -\\mu_{BL}^\\top w + \\frac{\\delta}{2}\\, w^\\top \\Sigma_{BL} w

    Subject to:
      * :math:`\\sum_i w_i = 1`
      * :math:`w_i \\geq \\texttt{min\\_weight}` for all *i*
      * :math:`w_i \\leq \\texttt{max\\_weight}` for all *i*

    Parameters
    ----------
    mu_bl : np.ndarray, shape (N,)
        BL posterior expected returns.
    cov_bl : np.ndarray, shape (N, N)
        BL posterior covariance matrix.
    assets : list[str]
        Ordered asset names (used for the output Series index).
    delta : float
        Risk-aversion coefficient (same value used throughout the model).
    min_weight : float
        Minimum allocation per asset (e.g. 0.05 = 5%).
    max_weight : float
        Maximum allocation per asset (e.g. 0.40 = 40%).

    Returns
    -------
    weights : pd.Series
        Optimal weights indexed by ``assets``.  Falls back to equal weight
        if SLSQP fails to converge.

    Notes
    -----
    The lower bound ``min_weight`` must satisfy  N * min_weight ≤ 1,
    otherwise the feasible region is empty.  This is checked silently and
    the bound is relaxed to 0 if violated.
    """
    n = len(assets)
    w0 = np.ones(n) / n

    # Safety check: if min_weight * n > 1, the constraint is infeasible
    effective_min = min_weight if (min_weight * n <= 1.0) else 0.0
    # Also ensure max_weight >= 1/n so equal weight is always feasible
    effective_max = max(max_weight, 1.0 / n)

    bounds = [(effective_min, effective_max)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_utility(w: np.ndarray) -> float:
        return -(mu_bl @ w) + (delta / 2.0) * (w @ cov_bl @ w)

    def neg_utility_grad(w: np.ndarray) -> np.ndarray:
        return -mu_bl + delta * (cov_bl @ w)

    res = minimize(
        neg_utility,
        w0,
        jac=neg_utility_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )

    if res.success:
        weights = np.maximum(res.x, 0.0)
        weights /= weights.sum()
        return pd.Series(weights, index=assets)

    # Graceful fallback to equal weight
    return pd.Series(w0, index=assets)


# ---------------------------------------------------------------------------
# Internal full pipeline helper
# ---------------------------------------------------------------------------

def _run_bl_pipeline(
    returns_window: pd.DataFrame,
    views: list[ViewDict],
    w_mkt: pd.Series | None,
    delta: float,
    tau: float,
    min_weight: float,
    max_weight: float,
    view_confidences: list[float] | None = None,
) -> pd.Series:
    """Execute the full BL pipeline on a single training window.

    This is the inner function called by the strategy factory. It:
      1. Computes the sample covariance.
      2. Resolves market weights (equal weight if ``w_mkt`` is None).
      3. Builds implied equilibrium returns.
      4. Constructs P, Q, Ω from ``views``.
      5. Computes BL posterior.
      6. Optimises and returns portfolio weights.

    On any unrecoverable numerical error, falls back to equal weight.

    Parameters
    ----------
    returns_window : pd.DataFrame
        Training window of log returns (rows = periods, cols = assets).
    views : list[dict]
        User-supplied view list.
    w_mkt : pd.Series or None
        Market-cap weights.  If None, uses equal weight as the prior.
    delta : float
        Risk-aversion coefficient.
    tau : float
        Prior uncertainty scalar.
    min_weight : float
        Minimum asset weight.
    max_weight : float
        Maximum asset weight.

    Returns
    -------
    pd.Series
        Portfolio weights indexed by asset name.
    """
    assets = list(returns_window.columns)
    n = len(assets)
    equal_w = pd.Series(1.0 / n, index=assets)

    # --- Covariance (monthly → annualised) -----------------------------------
    # CRITICAL: user-supplied views are in annualised units (e.g. 0.10 = 10%/yr).
    # The covariance computed on monthly log-returns is on a monthly scale
    # (~0.001), making Q/π ≈ 30–40×.  That completely dominates the posterior
    # and collapses the allocation to a corner solution (100% in one asset).
    #
    # Fix: annualise Σ by multiplying by 12 (periods per year for monthly data)
    # so that π = δ·Σ_annual·w is also annualised (~0.03–0.05) and on the same
    # scale as Q (~0.06–0.10).  Every downstream quantity (P, Q, Ω, μ_BL,
    # Σ_BL) then lives in consistent annual units.
    cov_df = returns_window.cov()
    cov = cov_df.values.astype(float) * 12   # ← annualise

    # Guard: singular covariance
    if np.linalg.matrix_rank(cov) < n:
        return equal_w

    # --- Market weights ------------------------------------------------------
    if w_mkt is None:
        w_arr = equal_w.values
    else:
        # Align to current asset universe; fill missing assets with 0
        aligned = w_mkt.reindex(assets).fillna(0.0)
        total = aligned.sum()
        w_arr = (aligned / total).values if total > 1e-12 else equal_w.values

    # --- Implied equilibrium returns (now annualised) ------------------------
    pi = implied_equilibrium_returns(cov, w_arr, delta=delta)

    # --- Filter views to assets in the current universe ----------------------
    available = set(assets)
    active_views: list[ViewDict] = []
    active_confidences: list[float] | None = None
    for i, v in enumerate(views):
        vtype = v.get("type", "").lower()
        if vtype == "absolute" and v.get("asset") in available:
            active_views.append(v)
            if view_confidences is not None:
                if active_confidences is None:
                    active_confidences = []
                active_confidences.append(
                    view_confidences[i] if i < len(view_confidences) else 0.5
                )
        elif vtype == "relative" and (
            v.get("asset_long") in available
            and v.get("asset_short") in available
        ):
            active_views.append(v)
            if view_confidences is not None:
                if active_confidences is None:
                    active_confidences = []
                active_confidences.append(
                    view_confidences[i] if i < len(view_confidences) else 0.5
                )

    # If no views survive, fall back to Minimum Variance.
    # NOTE: _mvo_from_pi with equal-weight prior always returns equal weight
    # by construction (pi = delta * Sigma * w_eq → w* = w_eq). Minimum
    # Variance is a more informative fallback when no views are provided.
    if not active_views:
        return _min_var_fallback(cov, assets, min_weight, max_weight)

    # --- View matrices -------------------------------------------------------
    try:
        P, Q, Omega = build_view_matrices(
            active_views, assets, cov, tau=tau, confidences=active_confidences
        )
    except (ValueError, np.linalg.LinAlgError):
        return equal_w

    # --- BL posterior --------------------------------------------------------
    try:
        mu_bl, cov_bl = bl_posterior(pi, cov, P, Q, Omega, tau=tau)
    except np.linalg.LinAlgError:
        return equal_w

    # --- Optimisation --------------------------------------------------------
    return bl_optimal_weights(
        mu_bl, cov_bl, assets,
        delta=delta,
        min_weight=min_weight,
        max_weight=max_weight,
    )


def _min_var_fallback(
    cov: np.ndarray,
    assets: list[str],
    min_weight: float,
    max_weight: float,
) -> pd.Series:
    """Minimum Variance portfolio used when no BL views are provided.

    The implied-equilibrium MVO (π-based) trivially returns the equal-weight
    prior when w_mkt is equal-weight — this is mathematically guaranteed by
    the reverse-optimisation construction.  Minimum Variance is a more
    informative fallback because it explicitly minimises portfolio risk
    regardless of expected returns.
    """
    n = len(assets)
    w0 = np.ones(n) / n
    effective_min = min_weight if (min_weight * n <= 1.0) else 0.0
    effective_max = max(max_weight, 1.0 / n)
    bounds = [(effective_min, effective_max)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        lambda w: w @ cov @ w,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )
    if res.success:
        wts = np.maximum(res.x, 0.0)
        return pd.Series(wts / wts.sum(), index=assets)
    return pd.Series(w0, index=assets)


def _mvo_from_pi(
    pi: np.ndarray,
    cov: np.ndarray,
    assets: list[str],
    delta: float,
    min_weight: float,
    max_weight: float,
) -> pd.Series:
    """Simple mean-variance optimiser using π as expected returns (no views).

    Parameters
    ----------
    pi : np.ndarray, shape (N,)
        Implied equilibrium excess returns.
    cov : np.ndarray, shape (N, N)
        Covariance matrix.
    assets : list[str]
        Asset names.
    delta : float
        Risk aversion.
    min_weight : float
        Minimum weight per asset.
    max_weight : float
        Maximum weight per asset.

    Returns
    -------
    pd.Series
        Portfolio weights.
    """
    # Reuse the same optimiser with no view correction
    return bl_optimal_weights(
        pi, cov, assets,
        delta=delta,
        min_weight=min_weight,
        max_weight=max_weight,
    )


# ---------------------------------------------------------------------------
# Risk-budget constraint helper (private)
# ---------------------------------------------------------------------------

def _apply_risk_budget_constraint(
    weights: pd.Series,
    cov: np.ndarray,
    max_risk_budget: float,
    min_weight: float,
    max_weight: float,
) -> pd.Series:
    """Enforce a per-asset Marginal Risk Contribution (MRC) cap.

    Computes MRC_i = w_i * (Σw)_i / sqrt(wᵀΣw) for each asset.
    If all MRC_i ≤ max_risk_budget, returns weights unchanged.
    Otherwise runs a secondary SLSQP optimisation that minimises
    sum((MRC_i − target)²) subject to sum(w)=1 and the original
    min/max bounds.  Falls back to original weights on failure.

    Parameters
    ----------
    weights : pd.Series
        Current portfolio weights (indexed by asset name).
    cov : np.ndarray, shape (N, N)
        Asset covariance matrix from the training window.
    max_risk_budget : float
        Maximum fraction of total portfolio volatility any single
        asset may contribute (e.g. 0.35 = 35%).
    min_weight : float
        Per-asset lower bound (same as used in the BL optimisation).
    max_weight : float
        Per-asset upper bound (same as used in the BL optimisation).

    Returns
    -------
    pd.Series
        Adjusted weights satisfying the risk-budget constraint, or the
        original weights if the constraint is already met or optimisation
        fails.
    """
    w = weights.values.copy()
    n = len(w)

    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return weights

    port_vol = np.sqrt(port_var)
    mrc = w * (cov @ w) / port_vol  # shape (N,)

    if float(mrc.max()) <= max_risk_budget:
        return weights  # constraint already satisfied

    # Target: equal risk contribution (1/N each)
    target = 1.0 / n

    def objective(w_opt: np.ndarray) -> float:
        pv = float(w_opt @ cov @ w_opt)
        if pv <= 0:
            return 1e10
        mrc_opt = w_opt * (cov @ w_opt) / np.sqrt(pv)
        return float(np.sum((mrc_opt - target) ** 2))

    effective_min = min_weight if (min_weight * n <= 1.0) else 0.0
    effective_max = max(max_weight, 1.0 / n)
    bounds = [(effective_min, effective_max)] * n
    constraints = [{"type": "eq", "fun": lambda w_opt: np.sum(w_opt) - 1.0}]

    res = minimize(
        objective,
        w,  # warm-start from BL weights
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )

    if res.success:
        new_w = np.maximum(res.x, 0.0)
        new_w /= new_w.sum()
        return pd.Series(new_w, index=weights.index)

    return weights  # fallback to original on optimisation failure


# ---------------------------------------------------------------------------
# Step 5 — Strategy Factory
# ---------------------------------------------------------------------------

def make_bl_strategy(
    views: list[ViewDict],
    w_mkt: pd.Series | None = None,
    delta: float = 2.5,
    tau: float = 0.05,
    min_weight: float = 0.05,
    max_weight: float = 0.40,
    max_risk_budget: float | None = None,
    view_confidences: list[float] | None = None,
) -> Callable[[pd.DataFrame], pd.Series]:
    """Factory that returns a Black-Litterman strategy compatible with the engine.

    The returned callable matches the engine interface::

        weights_fn(returns_window: pd.DataFrame) -> pd.Series

    Parameters
    ----------
    views : list[dict]
        List of view dictionaries.  Two formats are supported:

        Absolute view::

            {"type": "absolute", "asset": "SPY", "return": 0.07}

        Relative view::

            {"type": "relative",
             "asset_long": "SPY",
             "asset_short": "TLT",
             "return": 0.03}

    w_mkt : pd.Series or None
        Market-capitalisation weights indexed by ticker.  Normalised
        internally.  Pass ``None`` to use equal weight as the BL prior.
    delta : float
        Risk-aversion coefficient (λ in some texts).  Controls the
        curvature of the utility function.  Typical range: 1–4.
    tau : float
        Prior uncertainty scalar.  Small τ (e.g. 0.01–0.10) means high
        confidence in the equilibrium prior relative to sample estimates.
    min_weight : float
        Minimum allocation per asset.  Must satisfy N × min_weight ≤ 1,
        otherwise silently relaxed to 0.
    max_weight : float
        Maximum allocation per asset.  Enforces concentration limits.
    max_risk_budget : float or None
        Maximum fraction of total portfolio volatility (Marginal Risk
        Contribution) that any single asset may contribute.
        ``None`` (default) disables this constraint — exact current behaviour.
        Example: ``0.35`` caps each asset at 35% of portfolio volatility.
        After the BL optimisation, if any asset exceeds this budget a
        secondary SLSQP rebalances risk contributions toward equal
        contribution (1/N target) while respecting min/max weight bounds.
        Falls back to the original BL weights if the secondary optimisation
        fails.
    view_confidences : list[float] or None
        Per-view confidence levels in [0, 1].  Must be the same length as
        ``views``.  ``None`` (default) uses 0.5 for every view, which is
        identical to the standard He-Litterman proportional uncertainty.
        Higher values shift the BL posterior toward the view; lower values
        keep the posterior close to the equilibrium prior.

    Returns
    -------
    Callable[[pd.DataFrame], pd.Series]
        A strategy function that the walk-forward engine can call directly.

    Examples
    --------
    >>> bl = make_bl_strategy(
    ...     views=[
    ...         {"type": "absolute",  "asset": "SPY",               "return": 0.07},
    ...         {"type": "relative",  "asset_long": "SPY",
    ...          "asset_short": "TLT",                              "return": 0.03},
    ...     ],
    ...     w_mkt=None,   # equal-weight prior
    ...     delta=2.5,
    ...     tau=0.05,
    ...     min_weight=0.05,
    ...     max_weight=0.40,
    ...     max_risk_budget=0.35,   # no asset > 35% of portfolio vol
    ... )
    >>> # Register in STRATEGIES dict from engine.py
    >>> # STRATEGIES["Black-Litterman"] = bl
    """
    def _strategy(returns_window: pd.DataFrame, **kwargs) -> pd.Series:
        weights = _run_bl_pipeline(
            returns_window=returns_window,
            views=views,
            w_mkt=w_mkt,
            delta=delta,
            tau=tau,
            min_weight=min_weight,
            max_weight=max_weight,
            view_confidences=view_confidences,
        )
        if max_risk_budget is not None:
            cov = returns_window.cov().values.astype(float)
            weights = _apply_risk_budget_constraint(
                weights, cov, max_risk_budget, min_weight, max_weight
            )
        return weights

    _strategy.__name__ = "black_litterman"
    _strategy.__qualname__ = "black_litterman"
    return _strategy


# ---------------------------------------------------------------------------
# Step 7 — Comparison Utility: Black-Litterman vs. Markowitz
# ---------------------------------------------------------------------------

def _markowitz_weights(
    returns_window: pd.DataFrame,
    delta: float = 2.5,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """Classic mean-variance (Markowitz) weights using sample moments.

    Objective:  maximise  μ' w − (δ/2) w' Σ w
                subject to sum(w) = 1, min_weight ≤ w_i ≤ max_weight

    Parameters
    ----------
    returns_window : pd.DataFrame
        Training window of log returns.
    delta : float
        Risk-aversion coefficient.
    min_weight : float
        Minimum allocation per asset.
    max_weight : float
        Maximum allocation per asset.

    Returns
    -------
    pd.Series
        Markowitz weights indexed by asset name.
    """
    assets = list(returns_window.columns)
    n = len(assets)
    mu = returns_window.mean().values
    cov = returns_window.cov().values
    w0 = np.ones(n) / n

    effective_min = min_weight if (min_weight * n <= 1.0) else 0.0
    effective_max = max(max_weight, 1.0 / n)

    bounds = [(effective_min, effective_max)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_utility(w: np.ndarray) -> float:
        return -(mu @ w) + (delta / 2.0) * (w @ cov @ w)

    def neg_utility_grad(w: np.ndarray) -> np.ndarray:
        return -mu + delta * (cov @ w)

    res = minimize(
        neg_utility,
        w0,
        jac=neg_utility_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )
    if res.success:
        wts = np.maximum(res.x, 0.0)
        return pd.Series(wts / wts.sum(), index=assets)
    return pd.Series(w0, index=assets)


def compare_bl_vs_markowitz(
    returns_window: pd.DataFrame,
    views: list[ViewDict],
    w_mkt: pd.Series | None = None,
    delta: float = 2.5,
    tau: float = 0.05,
    min_weight: float = 0.05,
    max_weight: float = 0.40,
) -> pd.DataFrame:
    """Side-by-side comparison of Black-Litterman vs. Markowitz portfolios.

    Runs both allocation models on the same returns window and returns a
    summary DataFrame that includes implied equilibrium returns and BL
    posterior returns for interpretability.

    Parameters
    ----------
    returns_window : pd.DataFrame
        Training window of log returns (rows = periods, cols = assets).
    views : list[dict]
        List of view dictionaries in the standard format.
    w_mkt : pd.Series or None
        Market-cap weights.  If None, equal weight prior is used.
    delta : float
        Risk-aversion coefficient.
    tau : float
        Prior uncertainty scalar.
    min_weight : float
        Minimum allocation per asset (applied to both strategies).
    max_weight : float
        Maximum allocation per asset (applied to both strategies).

    Returns
    -------
    pd.DataFrame
        Columns:
          - ``asset``             : ticker symbol
          - ``markowitz_weight``  : classic MV optimal weight
          - ``bl_weight``         : Black-Litterman optimal weight
          - ``implied_return``    : π_i — implied equilibrium return
          - ``bl_return``         : μ_BL_i — BL posterior expected return

    Examples
    --------
    >>> df = compare_bl_vs_markowitz(
    ...     returns_window=ret_window,
    ...     views=[{"type": "absolute", "asset": "SPY", "return": 0.07}],
    ...     w_mkt=None,
    ... )
    >>> print(df.to_string(index=False))
    """
    assets = list(returns_window.columns)
    n = len(assets)
    equal_w_series = pd.Series(1.0 / n, index=assets)

    # Annualise covariance (consistent with _run_bl_pipeline fix)
    cov_df = returns_window.cov()
    cov = cov_df.values.astype(float) * 12   # ← annualise

    # --- Market weights ------------------------------------------------------
    if w_mkt is None:
        w_arr = equal_w_series.values
    else:
        aligned = w_mkt.reindex(assets).fillna(0.0)
        total = aligned.sum()
        w_arr = (aligned / total).values if total > 1e-12 else equal_w_series.values

    # --- Implied equilibrium returns (annualised) ----------------------------
    pi = implied_equilibrium_returns(cov, w_arr, delta=delta)

    # --- Markowitz weights ---------------------------------------------------
    mw = _markowitz_weights(
        returns_window,
        delta=delta,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    # --- Black-Litterman weights & posterior returns -------------------------
    bl_w = _run_bl_pipeline(
        returns_window=returns_window,
        views=views,
        w_mkt=w_mkt,
        delta=delta,
        tau=tau,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    # Reconstruct BL posterior returns for display
    active_views: list[ViewDict] = []
    available = set(assets)
    for v in views:
        vtype = v.get("type", "").lower()
        if vtype == "absolute" and v.get("asset") in available:
            active_views.append(v)
        elif vtype == "relative" and (
            v.get("asset_long") in available
            and v.get("asset_short") in available
        ):
            active_views.append(v)

    if active_views and np.linalg.matrix_rank(cov) == n:
        try:
            P, Q, Omega = build_view_matrices(active_views, assets, cov, tau=tau)
            mu_bl, _ = bl_posterior(pi, cov, P, Q, Omega, tau=tau)
        except (ValueError, np.linalg.LinAlgError):
            mu_bl = pi          # fallback: show equilibrium as BL return
    else:
        mu_bl = pi

    return pd.DataFrame(
        {
            "asset":            assets,
            "markowitz_weight": mw.values,
            "bl_weight":        bl_w.values,
            "implied_return":   pi,
            "bl_return":        mu_bl,
        }
    )
