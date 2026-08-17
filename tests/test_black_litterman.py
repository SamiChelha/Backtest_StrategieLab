"""
Test suite for the Black-Litterman module.
Run with: uv run python tests/test_black_litterman.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from strategy_lab.black_litterman import (
    implied_equilibrium_returns,
    build_view_matrices,
    bl_posterior,
    bl_optimal_weights,
    make_bl_strategy,
    compare_bl_vs_markowitz,
    _run_bl_pipeline,
)

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Fixture: realistic monthly log-returns (5 years, 3 assets)
# ─────────────────────────────────────────────────────────────────────────────
ASSETS = ["SPY", "AGG", "GLD"]
N = len(ASSETS)
T = 60  # 5 years monthly

# Simulate with known drift differences so strategies should diverge
mean_rets = np.array([0.008, 0.002, 0.004])   # SPY > GLD > AGG
cov_true = np.array([
    [ 0.0020,  0.0002, -0.0003],
    [ 0.0002,  0.0003,  0.0000],
    [-0.0003,  0.0000,  0.0008],
])
L = np.linalg.cholesky(cov_true)
raw = np.random.randn(T, N) @ L.T + mean_rets
returns_window = pd.DataFrame(raw, columns=ASSETS)

EQUAL_W = np.array([1/3, 1/3, 1/3])

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status}  {name}")
    if not condition:
        print(f"         ↳ {detail}")
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Implied equilibrium returns
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ TEST 1: Implied Equilibrium Returns ━━━")
cov = returns_window.cov().values
pi = implied_equilibrium_returns(cov, EQUAL_W, delta=2.5)
print(f"  π (annualized ×12): {dict(zip(ASSETS, (pi*12).round(4)))}")

check("π has correct shape", pi.shape == (N,), f"Got {pi.shape}")
check("π is positive for equity (SPY)", pi[0] > 0, f"π[SPY]={pi[0]:.5f}")
check("π is proportional to covariance × w_mkt",
      np.allclose(pi, 2.5 * cov @ EQUAL_W),
      "Reverse optimisation formula failed")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Equilibrium != Equal Weight (critical sanity check)
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ TEST 2: Equilibrium MVO != Equal Weight (no views) ━━━")
no_views_w = _run_bl_pipeline(
    returns_window,
    views=[],          # ← no views = should go through _mvo_from_pi
    w_mkt=None,
    delta=2.5,
    tau=0.05,
    min_weight=0.0,
    max_weight=1.0,
)
print(f"  No-view BL weights: {dict(zip(ASSETS, no_views_w.round(3).values))}")
print(f"  Equal weights:      {dict(zip(ASSETS, EQUAL_W.round(3)))}")
is_same = np.allclose(no_views_w.values, EQUAL_W, atol=0.01)
check("No-view BL ≠ Equal Weight (π-optimisation diverges)",
      not is_same,
      f"BL returned exact equal weight {no_views_w.values} — "
      "this means π optimisation collapses to equal weight when prior=equal.")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Views shift the posterior returns
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ TEST 3: Views Shift the BL Posterior ━━━")
views_bullish_spy = [{"type": "absolute", "asset": "SPY", "return": 0.12}]
P, Q, Omega = build_view_matrices(views_bullish_spy, ASSETS, cov, tau=0.05)
mu_bl, cov_bl = bl_posterior(pi, cov, P, Q, Omega, tau=0.05)
print(f"  Equilibrium π:  {dict(zip(ASSETS, pi.round(5)))}")
print(f"  BL posterior μ: {dict(zip(ASSETS, mu_bl.round(5)))}")
check("BL posterior ≠ equilibrium prior when views are applied",
      not np.allclose(pi, mu_bl, atol=1e-4),
      "Views had zero effect on the posterior")
check("SPY posterior return > SPY equilibrium (bullish view applied)",
      mu_bl[0] > pi[0],
      f"SPY: π={pi[0]:.5f}, μ_BL={mu_bl[0]:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: BL weights diverge from equal weight when bullish on SPY
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ TEST 4: BL Portfolio ≠ Equal Weight with Views ━━━")
bl_w_bullish = _run_bl_pipeline(
    returns_window,
    views=views_bullish_spy,
    w_mkt=None,
    delta=2.5, tau=0.05,
    min_weight=0.0, max_weight=1.0,
)
print(f"  BL weights (bullish SPY): {dict(zip(ASSETS, bl_w_bullish.round(3).values))}")
print(f"  Equal weights:            {dict(zip(ASSETS, EQUAL_W.round(3)))}")
check("BL weights differ from equal weight with bullish SPY view",
      not np.allclose(bl_w_bullish.values, EQUAL_W, atol=0.05),
      f"BL: {bl_w_bullish.values}, EW: {EQUAL_W}")
check("SPY is overweighted vs equal weight (bullish view reward)",
      bl_w_bullish["SPY"] > 1/N + 0.05,
      f"SPY weight = {bl_w_bullish['SPY']:.3f}, expected > {1/N+0.05:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Relative view (SPY outperforms AGG)
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ TEST 5: Relative View (SPY outperforms AGG by 5%) ━━━")
views_relative = [{"type": "relative", "asset_long": "SPY", "asset_short": "AGG", "return": 0.05}]
bl_w_rel = _run_bl_pipeline(
    returns_window,
    views=views_relative,
    w_mkt=None,
    delta=2.5, tau=0.05,
    min_weight=0.0, max_weight=1.0,
)
print(f"  BL weights (SPY vs AGG): {dict(zip(ASSETS, bl_w_rel.round(3).values))}")
check("SPY > AGG weight in relative view portfolio",
      bl_w_rel["SPY"] > bl_w_rel["AGG"],
      f"SPY={bl_w_rel['SPY']:.3f} AGG={bl_w_rel['AGG']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: BL vs Markowitz comparison function
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ TEST 6: compare_bl_vs_markowitz() ━━━")
comp = compare_bl_vs_markowitz(returns_window, views=views_bullish_spy, w_mkt=None)
print(comp.to_string(index=False))
check("Returns valid DataFrame", isinstance(comp, pd.DataFrame), str(type(comp)))
check("BL weights sum to 1.0", abs(comp["bl_weight"].sum() - 1.0) < 1e-6,
      f"Sum = {comp['bl_weight'].sum():.6f}")
check("Markowitz weights sum to 1.0", abs(comp["markowitz_weight"].sum() - 1.0) < 1e-6,
      f"Sum = {comp['markowitz_weight'].sum():.6f}")
check("BL and Markowitz differ (views should break symmetry)",
      not np.allclose(comp["bl_weight"].values, comp["markowitz_weight"].values, atol=0.02),
      "Weights are identical — views had no effect")
check("BL implied_return != bl_return (posterior shifted)",
      not np.allclose(comp["implied_return"].values, comp["bl_return"].values, atol=1e-4),
      "Posterior returns unchanged → views had no impact")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: Weight constraints are respected
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ TEST 7: Weight Constraints (min=0.10, max=0.60) ━━━")
bl_w_constrained = _run_bl_pipeline(
    returns_window,
    views=views_bullish_spy,
    w_mkt=None,
    delta=2.5, tau=0.05,
    min_weight=0.10, max_weight=0.60,
)
print(f"  Constrained weights: {dict(zip(ASSETS, bl_w_constrained.round(3).values))}")
check("All weights >= 0.10", all(bl_w_constrained >= 0.09),
      f"Min: {bl_w_constrained.min():.3f}")
check("All weights <= 0.60", all(bl_w_constrained <= 0.61),
      f"Max: {bl_w_constrained.max():.3f}")
check("Weights sum to 1.0", abs(bl_w_constrained.sum() - 1.0) < 1e-6,
      f"Sum={bl_w_constrained.sum():.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: Tau sensitivity (low tau = more weight on prior)
# ─────────────────────────────────────────────────────────────────────────────
print("\n━━━ TEST 8: Tau Sensitivity ━━━")
bl_low_tau = _run_bl_pipeline(
    returns_window, views=views_bullish_spy, w_mkt=None,
    delta=2.5, tau=0.005, min_weight=0.0, max_weight=1.0,
)
bl_high_tau = _run_bl_pipeline(
    returns_window, views=views_bullish_spy, w_mkt=None,
    delta=2.5, tau=0.5, min_weight=0.0, max_weight=1.0,
)
print(f"  Low  τ=0.005 weights: {dict(zip(ASSETS, bl_low_tau.round(3).values))}")
print(f"  High τ=0.500 weights: {dict(zip(ASSETS, bl_high_tau.round(3).values))}")
check("High τ gives more extreme SPY allocation (more confidence in views)",
      bl_high_tau["SPY"] >= bl_low_tau["SPY"],
      f"High τ SPY={bl_high_tau['SPY']:.3f}, Low τ SPY={bl_low_tau['SPY']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "━"*55)
print("ROOT CAUSE DIAGNOSIS")
print("━"*55)
if is_same:
    print("""
❌  CONFIRMED BUG: When no views are entered in the UI, BL
    falls back to _mvo_from_pi(). With equal-weight market prior
    and equal-weight prior returns (π = δ·Σ·w_eq), the MVO
    produces near-equal weights because the implied returns 
    barely differentiate assets when covariances are similar.

    SOLUTION: The UI must ALWAYS have at least 1 view for BL
    to produce meaningfully different allocations from Equal Weight.
    
    → Tell the user to add at least one View in the BL Settings!
    → Or: auto-generate views from historical momentum as default.
""")
else:
    print("""
✅  No collapse to equal weight detected in isolation tests.
    
    If you see equal-weight performance in the app, it is 
    because the UI is sending EMPTY VIEWS to the BL model.
    
    → Go to sidebar > Black-Litterman Settings
    → Click "➕ Add View" and enter at least ONE view
    → Example: Absolute view on SPY = +8% expected return
""")
print("━"*55)
