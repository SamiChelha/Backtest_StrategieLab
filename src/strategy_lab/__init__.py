"""
strategy_lab — Public API
==========================
This package is the core engine for the Strategy Lab backtester.

Clean public imports — use these instead of reaching into submodules:

    from strategy_lab import run_walk_forward, STRATEGIES, compute_summary
    from strategy_lab import download_prices, prices_to_log_returns, resample_to_month_end
    from strategy_lab import make_bl_strategy, compare_bl_vs_markowitz
    from strategy_lab import analyze_weight_constraints
    from strategy_lab import run_sensitivity_analysis
"""

from strategy_lab.engine import (
    run_walk_forward,
    STRATEGIES,
    analyze_weight_constraints,
    run_sensitivity_analysis,
)
from strategy_lab.metrics import compute_summary
from strategy_lab.data import download_prices, prices_to_log_returns, resample_to_month_end
from strategy_lab.black_litterman import make_bl_strategy, compare_bl_vs_markowitz

__all__ = [
    # Engine
    "run_walk_forward",
    "STRATEGIES",
    "analyze_weight_constraints",
    "run_sensitivity_analysis",
    # Metrics
    "compute_summary",
    # Data
    "download_prices",
    "prices_to_log_returns",
    "resample_to_month_end",
    # Black-Litterman
    "make_bl_strategy",
    "compare_bl_vs_markowitz",
]
