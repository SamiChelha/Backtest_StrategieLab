"""
User Interface — app/main.py
==============================
Streamlit dashboard for the Strategy Lab walk-forward backtester.
UI-only: all business logic is imported from strategy_lab.

Run:
    uv run streamlit run app/main.py
"""

import io
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from strategy_lab.data import download_prices, prices_to_log_returns, resample_to_month_end
from strategy_lab.engine import run_walk_forward, STRATEGIES, analyze_weight_constraints, run_sensitivity_analysis
from strategy_lab.metrics import compute_summary
from strategy_lab.black_litterman import make_bl_strategy, compare_bl_vs_markowitz

# ---------------------------------------------------------------------------
# Default configuration — single source of truth for all pre-filled values.
# Change any value here and it will propagate to every widget automatically.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    # Universe
    "tickers":          "SPY, EZU, GLD, TLT, AGG",
    "start_date":       datetime.date(2000, 1, 1),
    "end_date":         datetime.date.today(),
    # Strategy
    "strategy":         "Black-Litterman",
    "rf_pct":           3.0,
    "tc_bps":           10,
    # Black-Litterman parameters
    "bl_delta":         2.5,
    "bl_tau":           0.05,
    "bl_min_w":         0.05,
    "bl_max_w":         0.45,
    # Market-cap weights (one key per ticker: "mkt_<TICKER>")
    "use_custom_mkt":   True,
    "mkt_SPY":          0.50,
    "mkt_EZU":          0.18,
    "mkt_GLD":          0.09,
    "mkt_TLT":          0.13,
    "mkt_AGG":          0.10,
    # Overlay comparison strategies
    "compare_strats":   ["Momentum (12-1)", "Equal Weight", "Maximum Sharpe", "Inverse Volatility"],
    # BL views (absolute, annualised expected returns)
    "bl_views": [
        {"type": "absolute", "asset": "SPY", "return": 0.10, "confidence": 0.65},
        {"type": "absolute", "asset": "EZU", "return": 0.06, "confidence": 0.45},
        {"type": "absolute", "asset": "GLD", "return": 0.08, "confidence": 0.60},
        {"type": "absolute", "asset": "TLT", "return": 0.02, "confidence": 0.35},
        {"type": "absolute", "asset": "AGG", "return": 0.03, "confidence": 0.40},
    ],
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Strategy Lab", layout="wide", page_icon="💹")

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f1117;
        color: #e6edf3 !important;
    }

    /* Global text override — everything readable on dark */
    p, span, div, li, td, th, label, .stMarkdown, .stText {
        color: #e6edf3 !important;
    }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] span {
        color: #e6edf3 !important;
        font-size: 0.88rem;
    }
    /* Sidebar section headers (## text) */
    [data-testid="stSidebar"] h2 {
        color: #79c0ff !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 1.4rem;
        margin-bottom: 0.3rem;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(121,192,255,0.2);
    }
    /* Sidebar sub-headers (#### text) */
    [data-testid="stSidebar"] h4 {
        color: #adbac7 !important;
        font-size: 0.78rem !important;
        margin-top: 0.8rem;
    }
    /* Toggle / checkbox labels in sidebar */
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
    [data-testid="stSidebar"] [data-baseweb="checkbox"] span {
        color: #e6edf3 !important;
    }

    /* ── Metric cards ──────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(88,166,255,0.4) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        color: #adbac7 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* ── Run button ────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        transition: opacity 0.2s ease;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* ── Download buttons ──────────────────────────────────────────── */
    [data-testid="stDownloadButton"] > button {
        background: rgba(88,166,255,0.12) !important;
        border: 1px solid rgba(88,166,255,0.35) !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.45rem 1rem !important;
        width: 100% !important;
        transition: background 0.2s ease;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: rgba(88,166,255,0.22) !important;
        color: #ffffff !important;
    }

    /* ── Tabs ──────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #adbac7 !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(88,166,255,0.15) !important;
        color: #58a6ff !important;
    }

    /* ── Headings ──────────────────────────────────────────────────── */
    h1, h2 { color: #e6edf3 !important; font-weight: 700 !important; }
    /* h3 used as section sub-labels — keep readable */
    h3 {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #79c0ff !important;   /* bright blue — was near-invisible #adbac7 */
        margin-top: 1.4rem !important;
        margin-bottom: 0.5rem !important;
    }
    h4, h5 { color: #adbac7 !important; font-weight: 600 !important; }

    /* ── Table text ────────────────────────────────────────────────── */
    .stDataFrame td, .stDataFrame th {
        color: #e6edf3 !important;
    }

    /* ── Caption / small text ──────────────────────────────────────── */
    .stCaption, small {
        color: #8b949e !important;
    }

    /* ── Info / warning / success boxes ───────────────────────────── */
    .stAlert p { color: #e6edf3 !important; }
    [data-testid="stInfo"]    { border-left: 3px solid #58a6ff; }
    [data-testid="stWarning"] { border-left: 3px solid #ffa657; }
    [data-testid="stSuccess"] { border-left: 3px solid #3fb950; }

    /* ── Code blocks — light on dark, never dark-on-dark ──────────── */
    code, pre, .stCode, [data-testid="stCode"] {
        background-color: #161b22 !important;
        color: #79c0ff !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 6px !important;
    }
    pre code { border: none !important; }

    /* ── Expander headers ──────────────────────────────────────────── */
    [data-testid="stExpander"] summary {
        color: #e6edf3 !important;
        font-weight: 600;
    }
    [data-testid="stExpander"] summary:hover {
        color: #58a6ff !important;
    }
    [data-testid="stExpanderDetails"] {
        border-top: 1px solid rgba(255,255,255,0.06) !important;
    }

    /* ── Input widgets on dark background ─────────────────────────── */
    [data-testid="stDateInput"] input {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 6px !important;
    }
    [data-baseweb="calendar"],
    [data-baseweb="calendar"] * {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }
    [data-baseweb="calendar"] [aria-selected="true"] {
        background-color: #58a6ff !important;
        color: #0f1117 !important;
    }
    [data-baseweb="calendar"] button { color: #e6edf3 !important; }

    [data-testid="stSelectbox"] > div > div {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 6px !important;
    }
    [data-baseweb="select"] *, [data-baseweb="popover"] * {
        background-color: #21262d !important;
        color: #e6edf3 !important;
    }
    [data-baseweb="menu"] li:hover {
        background-color: rgba(88,166,255,0.15) !important;
    }

    [data-testid="stMultiSelect"] > div > div {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 6px !important;
    }
    [data-testid="stMultiSelect"] span { color: #e6edf3 !important; }

    [data-testid="stNumberInput"] input {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 6px !important;
    }
    [data-testid="stTextInput"] input {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 6px !important;
    }

    [data-testid="stRadio"] label  { color: #e6edf3 !important; }

    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: #8b949e !important;
    }
    /* Slider value tooltip */
    [data-testid="stSlider"] div[role="slider"] {
        background-color: #58a6ff !important;
    }

    [data-testid="stFileUploader"] {
        background-color: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small { color: #adbac7 !important; }
    [data-testid="stFileUploader"] button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 6px !important;
    }

    /* ── Divider ───────────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## 💹 Strategy Lab")
st.caption("Walk-Forward Portfolio Backtester — 100% Out-of-Sample")
st.divider()

# ---------------------------------------------------------------------------
# Session state defaults — initialised only once per browser session.
# All values come from DEFAULT_CONFIG so there is a single place to edit.
# ---------------------------------------------------------------------------
if "defaults_loaded" not in st.session_state:
    st.session_state["defaults_loaded"] = True
    for _k, _v in DEFAULT_CONFIG.items():
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Sidebar — Assets
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Assets")
ticker_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=st.session_state["tickers"],
    key="tickers",
    help="Any Yahoo Finance ticker. E.g: SPY, AGG, GLD, BTC-USD, AAPL"
)
selected_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

benchmark_ticker = st.sidebar.text_input("Benchmark (Beta & IR)", value="SPY")

data_source = st.sidebar.radio(
    "Data source",
    ["📡 Yahoo Finance", "📂 Upload CSV"],
    index=0,
)
if data_source == "📡 Yahoo Finance":
    start_date = st.sidebar.date_input(
        "Start",
        value=st.session_state["start_date"],
        min_value=datetime.date(1980, 1, 1),
        max_value=datetime.date.today(),
        key="start_date",
    )
    end_date = st.sidebar.date_input(
        "End",
        value=st.session_state["end_date"],
        min_value=datetime.date(1980, 1, 1),
        max_value=datetime.date.today(),
        key="end_date",
    )
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload prices CSV", type=["csv"])
    start_date, end_date = None, None

# ---------------------------------------------------------------------------
# Sidebar — Strategy
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Strategy")
strategy = st.sidebar.selectbox(
    "Strategy",
    options=list(STRATEGIES.keys()),
    index=list(STRATEGIES.keys()).index(st.session_state.get("strategy", "Black-Litterman")),
    key="strategy",
)
min_train_months = st.sidebar.number_input("Min. training months", 6, 60, 12,
    help="Warm-up period before first OOS prediction.")
rebalance_months = st.sidebar.number_input("Rebalance every (months)", 1, 24, 1)

# ---------------------------------------------------------------------------
# Sidebar — Walk-Forward
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Walk-Forward")
wf_method = st.sidebar.radio(
    "Window", ["expanding", "rolling"],
    help="Expanding: train on all history (rigorous). Rolling: fixed lookback."
)
transaction_cost_bps = st.sidebar.slider(
    "Transaction cost (bps)", 0, 50,
    value=st.session_state["tc_bps"],
    key="tc_bps",
    help="Applied to turnover at each rebalancing date.",
)

rebalancing_type = st.sidebar.selectbox(
    "Rebalancing mode",
    options=["monthly", "drift_band", "yearly"],
    index=0,
    help=(
        "**monthly** — rebalance every walk-forward step (default).  \n"
        "**drift_band** — rebalance only when any asset deviates > threshold from target w*.  \n"
        "**yearly** — rebalance every 12 months only."
    ),
)
drift_threshold = st.sidebar.slider(
    "Drift threshold",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    disabled=(rebalancing_type != "drift_band"),
    help="Max allowed deviation from target w* before rebalancing (drift_band mode only).",
)

# ---------------------------------------------------------------------------
# Sidebar — Weight Constraints (ALL strategies)
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 📊 Weight Constraints")
wc_min = st.sidebar.slider(
    "Min weight per asset", 0.0, 0.25, 0.0, step=0.01,
    help="Per-asset lower bound.  Enforced by optimisation strategies (MinVar, MaxSharpe, RiskParity)."
)
wc_max = st.sidebar.slider(
    "Max weight per asset", 0.25, 1.0, 1.0, step=0.05,
    help="Per-asset upper bound.  Prevents over-concentration."
)

# ---------------------------------------------------------------------------
# Sidebar — Black-Litterman Settings (only when BL selected)
# ---------------------------------------------------------------------------
bl_delta = st.session_state.get("bl_delta", DEFAULT_CONFIG["bl_delta"])
bl_tau = st.session_state.get("bl_tau", DEFAULT_CONFIG["bl_tau"])
bl_min_w = st.session_state.get("bl_min_w", DEFAULT_CONFIG["bl_min_w"])
bl_max_w = st.session_state.get("bl_max_w", DEFAULT_CONFIG["bl_max_w"])
use_custom_mkt = st.session_state.get("use_custom_mkt", DEFAULT_CONFIG["use_custom_mkt"])
custom_mkt_weights: dict[str, float] = {}

is_bl = strategy == "Black-Litterman"

if is_bl:
    st.sidebar.markdown("## ⚙️ Black-Litterman Settings")

    # ── Global BL parameters ──────────────────────────────────────────
    bl_delta = st.sidebar.slider(
        "δ (risk aversion)", 0.5, 5.0,
        value=st.session_state["bl_delta"],
        step=0.5, key="bl_delta",
        help="Higher δ → more risk-averse → equilibrium returns are higher.",
    )
    bl_tau = st.sidebar.slider(
        "τ (uncertainty scalar)", 0.01, 0.20,
        value=st.session_state["bl_tau"],
        step=0.01, key="bl_tau",
        help="Smaller τ → more weight on the equilibrium prior.",
    )

    # ── BL weight constraints ─────────────────────────────────────────
    bl_min_w = st.sidebar.slider(
        "BL min weight", 0.0, 0.20,
        value=st.session_state["bl_min_w"],
        step=0.01, key="bl_min_w",
        help="Per-asset lower bound in BL optimisation.",
    )
    bl_max_w = st.sidebar.slider(
        "BL max weight", 0.20, 1.0,
        value=st.session_state["bl_max_w"],
        step=0.05, key="bl_max_w",
        help="Per-asset upper bound in BL optimisation.",
    )

    # ── Market-cap weights (optional) ─────────────────────────────────
    use_custom_mkt = st.sidebar.toggle(
        "Use custom market cap weights",
        value=st.session_state["use_custom_mkt"],
        key="use_custom_mkt",
        help="If disabled, BL uses equal weights as the equilibrium prior.",
    )
    if use_custom_mkt and selected_tickers:
        st.sidebar.caption("Enter market-cap weights (must sum to 1.0):")
        for t in selected_tickers:
            custom_mkt_weights[t] = st.sidebar.number_input(
                f"w({t})", 0.0, 1.0,
                value=st.session_state.get(f"mkt_{t}", round(1.0 / len(selected_tickers), 4)),
                step=0.01, key=f"mkt_{t}",
            )
        mkt_sum = sum(custom_mkt_weights.values())
        if abs(mkt_sum - 1.0) > 0.01:
            st.sidebar.warning(f"⚠️ Weights sum to {mkt_sum:.4f}, not 1.0")

    # ── Views builder ─────────────────────────────────────────────────
    st.sidebar.markdown("#### 🎯 Views")
    st.sidebar.caption(
        "Views are **mandatory** for BL to differ from equal-weight. "
        "Add at least one absolute or relative view."
    )

    MAX_VIEWS = 6
    views_list = st.session_state["bl_views"]

    # ── No-views warning ──────────────────────────────────────────────
    if len(views_list) == 0:
        st.sidebar.error(
            "⚠️ **No views defined.** Without views, Black-Litterman "
            "collapses to equal weight (by mathematical construction). "
            "Add at least one view below."
        )

    # Add view button
    if len(views_list) < MAX_VIEWS:
        if st.sidebar.button("➕ Add View"):
            default_asset = selected_tickers[0] if selected_tickers else ""
            views_list.append({"type": "absolute", "asset": default_asset, "return": 0.08})
            st.rerun()

    # Render each view
    indices_to_remove: list[int] = []
    for idx, view in enumerate(views_list):
        with st.sidebar.container():
            st.sidebar.markdown(f"---\n**View {idx + 1}**")
            c1, c2 = st.sidebar.columns([3, 1])
            vtype = c1.selectbox(
                "Type", ["Absolute", "Relative"],
                index=0 if view.get("type", "absolute") == "absolute" else 1,
                key=f"vtype_{idx}",
            )
            if c2.button("🗑️", key=f"vdel_{idx}"):
                indices_to_remove.append(idx)

            if vtype == "Absolute":
                asset_abs = st.sidebar.selectbox(
                    "Asset", selected_tickers, key=f"vasset_{idx}",
                    index=min(idx, len(selected_tickers) - 1),
                )
                ret_abs = st.sidebar.number_input(
                    "Expected annual return", -0.50, 0.50,
                    value=float(view.get("return", 0.08)),
                    step=0.01, key=f"vret_{idx}",
                )
                confidence = st.sidebar.slider(
                    f"Confidence — View {idx + 1}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(view.get("confidence", 0.5)),
                    step=0.05,
                    key=f"conf_{idx}",
                    help=(
                        "0.0 = no conviction (BL stays at equilibrium prior)  |  "
                        "1.0 = total certainty (BL = your view exactly)  |  "
                        "0.5 = moderate confidence (default)"
                    ),
                )
                views_list[idx] = {
                    "type": "absolute", "asset": asset_abs, "return": ret_abs,
                    "confidence": confidence,
                }
            else:
                if len(selected_tickers) < 2:
                    st.sidebar.warning("Need 2+ tickers for relative views.")
                else:
                    al = st.sidebar.selectbox(
                        "Long asset", selected_tickers, key=f"vlong_{idx}",
                    )
                    short_options = [t for t in selected_tickers if t != al]
                    ash = st.sidebar.selectbox(
                        "Short asset", short_options, key=f"vshort_{idx}",
                    )
                    ret_rel = st.sidebar.number_input(
                        "Outperformance", -0.50, 0.50,
                        value=float(view.get("return", 0.03)),
                        step=0.01, key=f"vretrel_{idx}",
                    )
                    confidence = st.sidebar.slider(
                        f"Confidence — View {idx + 1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(view.get("confidence", 0.5)),
                        step=0.05,
                        key=f"conf_{idx}",
                        help=(
                            "0.0 = no conviction (BL stays at equilibrium prior)  |  "
                            "1.0 = total certainty (BL = your view exactly)  |  "
                            "0.5 = moderate confidence (default)"
                        ),
                    )
                    views_list[idx] = {
                        "type": "relative",
                        "asset_long": al,
                        "asset_short": ash,
                        "return": ret_rel,
                        "confidence": confidence,
                    }

    # Process removals
    if indices_to_remove:
        for rm_idx in sorted(indices_to_remove, reverse=True):
            views_list.pop(rm_idx)
        st.rerun()

    st.session_state["bl_views"] = views_list

    # ── Reset to a single default view (not empty!) ────────────────────
    if st.sidebar.button("🔄 Reset BL to defaults"):
        default_asset = selected_tickers[0] if selected_tickers else "SPY"
        st.session_state["bl_views"] = [
            {"type": "absolute", "asset": default_asset, "return": 0.08}
        ]
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — Comparison & Risk-Free
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Comparison")
_compare_opts = [s for s in STRATEGIES.keys() if s != strategy]
# Remove any stale selections that are no longer valid options (e.g. after strategy change)
if "compare_strats" in st.session_state:
    st.session_state["compare_strats"] = [
        s for s in st.session_state["compare_strats"] if s in _compare_opts
    ]
compare_strategies = st.sidebar.multiselect(
    "Overlay strategies",
    options=_compare_opts,
    default=st.session_state["compare_strats"],
    key="compare_strats",
)

st.sidebar.markdown("## Risk-Free Rate")
rf_annual_pct = st.sidebar.number_input(
    "Risk-free rate (% per year)",
    min_value=0.0, max_value=20.0,
    value=st.session_state["rf_pct"],
    step=0.1, key="rf_pct",
    help="Annual risk-free rate used in Sharpe & Sortino (e.g. 4.5 for 4.5%). Set 0 to ignore.",
)

run = st.sidebar.button("▶  Run backtest")


def _load_benchmark(ticker, start, end):
    if not ticker.strip():
        return None
    try:
        bp = download_prices([ticker.strip().upper()], str(start), str(end))
        return prices_to_log_returns(resample_to_month_end(bp)).iloc[:, 0].dropna()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.2)",
    font=dict(family="Inter", size=12, color="#8b949e"),
    margin=dict(l=10, r=10, t=40, b=10),
)

COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff"]


def _hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    """Convert a #RRGGBB hex string to a valid rgba() string for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _wealth_fig(wealth, strategy, comparison_wealth):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wealth.index, y=wealth.values,
        mode="lines", name=strategy,
        line=dict(color=COLORS[0], width=2.5),
        hovertemplate="%{y:.3f}<extra>" + strategy + "</extra>",
    ))
    for i, (name, w) in enumerate(comparison_wealth.items()):
        fig.add_trace(go.Scatter(
            x=w.index, y=w.values,
            mode="lines", name=name,
            line=dict(color=COLORS[(i + 1) % len(COLORS)], width=1.5, dash="dash"),
            hovertemplate="%{y:.3f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        **DARK_LAYOUT, height=360,
        hovermode="x unified",
        yaxis_title="Wealth (start = 1.0)",
        legend=dict(
            orientation="h", y=1.08,
            font=dict(color="#e6edf3", size=11),  # white legend labels
        ),
    )
    return fig


def _drawdown_fig(dd):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", mode="lines",
        line=dict(color="#f78166", width=1),
        fillcolor="rgba(247,129,102,0.15)",
        name="Drawdown",
        hovertemplate="%{y:.1%}<extra></extra>",
    ))
    fig.update_layout(**DARK_LAYOUT, height=240,
                      hovermode="x unified",
                      yaxis_tickformat=".0%",
                      yaxis_title="Drawdown")
    return fig


def _weights_fig(wts_rebal, strategy):
    wts_long = wts_rebal.reset_index().melt(
        id_vars="index", var_name="Asset", value_name="Weight"
    ).rename(columns={"index": "Date"})
    wts_long["Date"] = wts_long["Date"].dt.strftime("%Y-%m")
    fig = px.bar(
        wts_long, x="Date", y="Weight", color="Asset",
        barmode="group", text_auto=".0%",
        color_discrete_sequence=COLORS,
        labels={"Weight": "Allocation"},
    )
    fig.update_traces(textposition="outside", textfont_size=9)
    fig.update_layout(**DARK_LAYOUT, height=360,
                      yaxis_tickformat=".0%", yaxis_range=[0, 1.2],
                      xaxis_tickangle=-45,
                      legend=dict(orientation="h", y=1.08))
    return fig


def _gantt_fig(folds_meta, max_folds=60):
    """Walk-forward timeline using px.timeline (correct Gantt implementation)."""
    step = max(1, len(folds_meta) // max_folds)
    sample = folds_meta.iloc[::step].copy().reset_index(drop=True)

    rows = []
    for _, r in sample.iterrows():
        label = r["test_date"].strftime("%Y-%m")
        rows.append(dict(
            Fold=label,
            Start=r["train_start"],
            End=r["train_end"],
            Phase="Training window",
        ))
        rows.append(dict(
            Fold=label,
            Start=r["test_date"],
            End=r["test_date"] + pd.offsets.MonthEnd(1),
            Phase="OOS Test",
        ))

    df = pd.DataFrame(rows)
    fig = px.timeline(
        df,
        x_start="Start", x_end="End",
        y="Fold", color="Phase",
        color_discrete_map={
            "Training window": "rgba(88,166,255,0.50)",
            "OOS Test":        "#f78166",
        },
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=10, color="#e6edf3"),
        gridcolor="rgba(255,255,255,0.05)",
    )
    fig.update_xaxes(
        tickfont=dict(size=10, color="#adbac7"),
        gridcolor="rgba(255,255,255,0.05)",
    )
    fig.update_layout(
        **DARK_LAYOUT,
        height=max(320, min(680, len(sample) * 10 + 60)),
        legend=dict(orientation="h", y=1.05, font=dict(color="#e6edf3")),
        xaxis_title="",
        yaxis_title="",
    )
    return fig, step


def _heatmap_fig(log_returns: pd.Series):
    """Monthly returns heatmap: rows = year, columns = Jan–Dec.

    Each cell shows the log-return converted to a simple % for readability.
    Colorscale: red (negative) → yellow (zero) → green (positive).
    Annotations always use high-contrast white text with a dark background
    shadow for legibility on any cell colour.
    """
    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    df = log_returns.to_frame(name="ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="sum")
    pivot = pivot.reindex(columns=range(1, 13))   # ensure all 12 months

    # Convert log-return → simple % for display
    pct = (np.exp(pivot) - 1) * 100

    n_years = len(pct)
    row_px   = 38   # fixed row height in pixels
    height   = max(280, n_years * row_px + 80)

    # ---- Build figure with named x-axis (month names) ----------------
    # Use the raw numeric matrix but label x-axis with month names.
    fig = px.imshow(
        pct.values,
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        aspect="auto",
        x=MONTH_NAMES,
        y=[str(y) for y in pct.index],
        zmin=-15,   # clip at ±15% for better colour contrast on typical returns
        zmax=15,
    )

    # ---- Annotations: always white with subtle dark bg ---------------
    annotations = []
    for ri in range(len(pct.index)):
        for ci in range(len(MONTH_NAMES)):
            val = pct.values[ri, ci]
            if np.isnan(val):
                continue
            # Choose white or dark text based on cell brightness
            # For RdYlGn: bright yellow at 0, dark red/green at extremes
            # → white always works best; use dark only for near-zero yellow cells
            txt_color = "#0d1117" if abs(val) < 4 else "#ffffff"
            annotations.append(dict(
                x=MONTH_NAMES[ci],
                y=str(pct.index[ri]),
                text=f"<b>{val:+.1f}%</b>",
                showarrow=False,
                font=dict(size=9, color=txt_color),
                xref="x", yref="y",
            ))

    # Merge DARK_LAYOUT with heatmap-specific overrides to avoid duplicate-kwarg crash
    # (DARK_LAYOUT already contains 'margin', so we must override it in the dict not as a kwarg)
    _hm_layout = {**DARK_LAYOUT, "margin": dict(l=10, r=10, t=50, b=10)}
    fig.update_layout(
        **_hm_layout,
        height=height,
        coloraxis_colorbar=dict(
            title="Rtn %",
            ticksuffix="%",
            thickness=12,
            len=0.8,
            tickfont=dict(color="#adbac7", size=10),
            title_font=dict(color="#adbac7"),
        ),
        annotations=annotations,
        xaxis_title="",
        yaxis_title="",
    )
    fig.update_xaxes(
        side="top",
        tickfont=dict(size=11, color="#e6edf3"),
        tickangle=0,
        fixedrange=True,
        gridcolor="rgba(255,255,255,0.05)",
    )
    fig.update_yaxes(
        tickfont=dict(size=11, color="#e6edf3"),
        autorange="reversed",
        fixedrange=True,
        gridcolor="rgba(255,255,255,0.05)",
    )
    return fig


# ---------------------------------------------------------------------------
# NEW plot helpers
# ---------------------------------------------------------------------------

def _stacked_area_fig(wts_df: pd.DataFrame):
    """Stacked area chart showing portfolio weight evolution over time."""
    fig = go.Figure()
    cols = [c for c in wts_df.columns if c != "Strategy"]
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=wts_df.index, y=wts_df[col],
            mode="lines", name=col,
            stackgroup="weights",
            line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
            fillcolor=_hex_to_rgba(COLORS[i % len(COLORS)], alpha=0.6),
            hovertemplate="%{y:.1%}<extra>" + col + "</extra>",
        ))
    fig.update_layout(
        **DARK_LAYOUT, height=350,
        yaxis_tickformat=".0%", yaxis_title="Weight",
        yaxis_range=[0, 1],
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def _turnover_fig(folds_meta: pd.DataFrame):
    """Turnover bar chart with mean line and annual annotation."""
    rebal = folds_meta[folds_meta["rebalanced"]].copy()
    if rebal.empty:
        return None
    rebal["turnover_pct"] = rebal["turnover"] * 100

    mean_to = rebal["turnover_pct"].mean()

    # Estimate annual turnover (turnovers per year × mean)
    n_rebal = len(rebal)
    span_years = ((rebal["test_date"].max() - rebal["test_date"].min()).days + 30) / 365.25
    annual_to = (n_rebal / max(span_years, 0.5)) * mean_to

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rebal["test_date"], y=rebal["turnover_pct"],
        marker_color="#58a6ff", opacity=0.7,
        name="Turnover",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=mean_to, line_dash="dash", line_color="#ffa657",
                  annotation_text=f"Mean: {mean_to:.1f}%",
                  annotation_font_color="#ffa657")
    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"<b>Annual turnover ≈ {annual_to:.0f}%</b>",
        showarrow=False,
        font=dict(color="#d2a8ff", size=12),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="#d2a8ff",
        borderwidth=1,
        borderpad=6,
    )
    fig.update_layout(**DARK_LAYOUT, height=280,
                      yaxis_title="Turnover (%)",
                      hovermode="x unified")
    return fig, annual_to


def _gross_vs_net_fig(
    returns: pd.DataFrame,
    strategy_name: str,
    min_train_months: int,
    method: str,
    rebalance_months: int,
    min_weight: float,
    max_weight: float,
    rebalancing_type: str = "monthly",
    drift_threshold: float = 0.05,
):
    """Gross vs Net cumulative return with shaded cost drag."""
    # Gross = 0 bps (same rebalancing mode as the main run)
    r_gross, _ = run_walk_forward(
        returns, strategy=strategy_name,
        min_train_months=min_train_months,
        method=method,
        transaction_cost_bps=0.0,
        rebalance_months=rebalance_months,
        min_weight=min_weight,
        max_weight=max_weight,
        rebalancing_type=rebalancing_type,
        drift_threshold=drift_threshold,
    )
    # Net already computed → we pass it in
    # Just compute gross wealth here
    if r_gross.empty:
        return None
    g_wealth = np.exp(r_gross["Strategy"].cumsum())
    return g_wealth


def _bl_returns_fig(assets, pi, mu_bl):
    """Horizontal bar chart: implied equilibrium (π) vs BL posterior (μ_BL)."""
    df = pd.DataFrame({
        "Asset": assets * 2,
        "Return": list(pi) + list(mu_bl),
        "Type": ["Implied Equilibrium (π)"] * len(assets) + ["BL Posterior (μ_BL)"] * len(assets),
    })
    fig = px.bar(
        df, y="Asset", x="Return", color="Type",
        orientation="h", barmode="group",
        color_discrete_map={
            "Implied Equilibrium (π)": "#58a6ff",
            "BL Posterior (μ_BL)": "#3fb950",
        },
    )
    fig.update_layout(
        **DARK_LAYOUT, height=max(250, len(assets) * 60),
        xaxis_tickformat=".2%",
        xaxis_title="Expected Return (annualised)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#e6edf3")),
        yaxis=dict(tickfont=dict(color="#e6edf3", size=12)),
        legend=dict(orientation="h", y=1.14, font=dict(color="#e6edf3", size=11)),
    )
    return fig


def _bl_weights_comparison_fig(comp_df: pd.DataFrame):
    """Side-by-side bar chart of Markowitz vs BL weights.

    Expects comp_df to be indexed by asset name (i.e. 'asset' column set as index).
    """
    assets = comp_df.index.tolist()
    df = pd.DataFrame({
        "Asset": assets * 2,
        "Weight": list(comp_df["markowitz_weight"]) + list(comp_df["bl_weight"]),
        "Model": ["Markowitz"] * len(assets) + ["Black-Litterman"] * len(assets),
    })
    fig = px.bar(
        df, x="Asset", y="Weight", color="Model",
        barmode="group", text_auto=".1%",
        color_discrete_map={
            "Markowitz": "#f78166",
            "Black-Litterman": "#3fb950",
        },
    )
    fig.update_traces(textposition="outside", textfont=dict(size=10, color="#e6edf3"))
    fig.update_layout(
        **DARK_LAYOUT, height=380,
        yaxis_tickformat=".0%", yaxis_range=[0, 1.15],
        xaxis=dict(tickfont=dict(color="#e6edf3", size=12)),
        yaxis_title="Allocation",
        legend=dict(orientation="h", y=1.10, font=dict(color="#e6edf3", size=11)),
    )
    return fig


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------

_PDF_BG      = "#0f1117"
_PDF_BLUE    = "#58a6ff"
_PDF_GREEN   = "#3fb950"
_PDF_RED     = "#f78166"
_PDF_PURPLE  = "#d2a8ff"
_PDF_ORANGE  = "#ffa657"
_PDF_TEXT    = "#e6edf3"
_PDF_SUBTEXT = "#8b949e"
_PDF_COLORS  = [_PDF_BLUE, _PDF_GREEN, _PDF_RED, _PDF_PURPLE, _PDF_ORANGE, "#79c0ff"]


def _generate_pdf_report(
    strategy: str,
    summary: dict,
    wts: pd.DataFrame,
    strat: pd.Series,
    folds_meta: pd.DataFrame,
    wealth: pd.Series,
    dd: pd.Series,
    comparison_wealth: dict,
    config: dict,
) -> bytes:
    """Generate a multi-page PDF report and return it as bytes.

    Pages
    -----
    1  Cover — dark blue, title, universe, period, method
    2  Performance Metrics — styled table
    3  Cumulative Wealth — matplotlib line chart
    4  Drawdown — red filled area chart
    5  Portfolio Weights Over Time — stacked area chart
    6  Portfolio Manager Interpretation (all strategies)
    """
    buf = io.BytesIO()

    with PdfPages(buf) as pdf:

        # ------------------------------------------------------------------
        # Page 1 — Cover
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("#1565C0")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("#1565C0")
        ax.axis("off")

        ax.text(0.5, 0.82, "STRATEGY LAB",
                ha="center", va="center", fontsize=28, fontweight="bold",
                color="white", transform=ax.transAxes)
        ax.text(0.5, 0.74, "PORTFOLIO ANALYSIS REPORT",
                ha="center", va="center", fontsize=16,
                color="rgba(255,255,255,0.85)" if False else "#bbdefb",
                transform=ax.transAxes)

        ax.add_patch(mpatches.FancyBboxPatch(
            (0.1, 0.44), 0.8, 0.22,
            boxstyle="round,pad=0.02",
            facecolor="rgba(0,0,0,0.0)" if False else "#0d47a1",
            edgecolor="#42a5f5", linewidth=1.5,
            transform=ax.transAxes,
        ))
        ax.text(0.5, 0.63, strategy,
                ha="center", va="center", fontsize=20, fontweight="bold",
                color="white", transform=ax.transAxes)

        tickers_str = " · ".join(config.get("tickers", []))
        ax.text(0.5, 0.555, f"Universe: {tickers_str}",
                ha="center", va="center", fontsize=12, color="#bbdefb",
                transform=ax.transAxes)
        ax.text(0.5, 0.505, f"Period: {config.get('start', '')} → {config.get('end', '')}",
                ha="center", va="center", fontsize=12, color="#bbdefb",
                transform=ax.transAxes)
        ax.text(0.5, 0.455, f"Generated: {datetime.date.today().strftime('%B %d, %Y')}",
                ha="center", va="center", fontsize=10, color="#90caf9",
                transform=ax.transAxes)

        ax.text(0.5, 0.10,
                "Walk-Forward  ·  Expanding Window  ·  100% Out-of-Sample",
                ha="center", va="center", fontsize=10, color="#90caf9",
                style="italic", transform=ax.transAxes)
        ax.text(0.5, 0.055,
                f"Transaction cost: {config.get('transaction_cost_bps', 10)} bps  "
                f"·  Method: {config.get('wf_method', 'expanding').capitalize()}",
                ha="center", va="center", fontsize=9, color="#90caf9",
                transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 2 — Performance Metrics Table
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor(_PDF_BG)
        ax.set_facecolor(_PDF_BG)
        ax.axis("off")

        ax.text(0.5, 0.96, "Performance Metrics",
                ha="center", va="top", fontsize=18, fontweight="bold",
                color=_PDF_TEXT, transform=ax.transAxes)
        ax.text(0.5, 0.915, strategy,
                ha="center", va="top", fontsize=12, color=_PDF_SUBTEXT,
                transform=ax.transAxes)

        def _fmt_val(key, val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            pct_keys = {"annualized_return", "max_drawdown", "total_return"}
            return f"{val:.2%}" if key in pct_keys else f"{val:.3f}"

        def _is_good(key, val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            good_high  = {"annualized_return", "sharpe", "sortino", "calmar",
                          "total_return", "information_ratio"}
            good_low   = {"max_drawdown"}
            if key in good_high:
                return val > 0
            if key in good_low:
                return val > -0.20
            return None

        rows = [
            ("Annualized Return",   "annualized_return"),
            ("Sharpe Ratio",        "sharpe"),
            ("Sortino Ratio",       "sortino"),
            ("Max Drawdown",        "max_drawdown"),
            ("Calmar Ratio",        "calmar"),
            ("Total Return",        "total_return"),
            ("Beta",                "beta"),
            ("Information Ratio",   "information_ratio"),
            ("OOS Months",          "n_months"),
        ]

        row_h  = 0.055
        y_start = 0.87
        col_x  = [0.08, 0.60]

        # Header
        for x, label in zip(col_x, ["Metric", "Value"]):
            ax.add_patch(plt.Rectangle(
                (x - 0.02, y_start - 0.005), 0.50, row_h,
                transform=ax.transAxes, color="#1565C0", zorder=1,
            ))
            ax.text(x, y_start + row_h * 0.35, label,
                    ha="left", va="center", fontsize=11, fontweight="bold",
                    color="white", transform=ax.transAxes, zorder=2)

        for r_idx, (label, key) in enumerate(rows):
            y = y_start - (r_idx + 1) * row_h
            bg = "#1a1f2e" if r_idx % 2 == 0 else "#141922"
            ax.add_patch(plt.Rectangle(
                (0.06, y - 0.004), 0.88, row_h,
                transform=ax.transAxes, color=bg, zorder=1,
            ))
            ax.text(col_x[0], y + row_h * 0.30, label,
                    ha="left", va="center", fontsize=10,
                    color=_PDF_TEXT, transform=ax.transAxes, zorder=2)

            val    = summary.get(key)
            v_str  = _fmt_val(key, val)
            good   = _is_good(key, val)
            v_col  = (_PDF_GREEN if good is True
                      else _PDF_RED if good is False
                      else _PDF_TEXT)
            ax.text(col_x[1], y + row_h * 0.30, v_str,
                    ha="left", va="center", fontsize=10, fontweight="bold",
                    color=v_col, transform=ax.transAxes, zorder=2)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 3 — Cumulative Wealth
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(_PDF_BG)
        ax.set_facecolor("rgba(0,0,0,0.2)" if False else "#161b22")
        ax.tick_params(colors=_PDF_SUBTEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        ax.plot(wealth.index, wealth.values,
                color=_PDF_BLUE, linewidth=2, label=strategy)
        for i, (cname, cw) in enumerate(comparison_wealth.items()):
            ax.plot(cw.index, cw.values,
                    color=_PDF_COLORS[(i + 1) % len(_PDF_COLORS)],
                    linewidth=1.5, linestyle="--", label=cname)

        ax.set_title("Cumulative Wealth — Out-of-Sample Walk-Forward",
                     color=_PDF_TEXT, fontsize=13, pad=12)
        ax.set_ylabel("Wealth (start = 1.0)", color=_PDF_SUBTEXT)
        ax.tick_params(axis="x", labelcolor=_PDF_SUBTEXT, rotation=30)
        ax.tick_params(axis="y", labelcolor=_PDF_SUBTEXT)
        ax.legend(facecolor="#21262d", edgecolor="#30363d",
                  labelcolor=_PDF_TEXT, fontsize=9)
        ax.grid(color="#21262d", linewidth=0.5)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 4 — Drawdown
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(_PDF_BG)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=_PDF_SUBTEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        ax.fill_between(dd.index, dd.values, 0,
                        color=_PDF_RED, alpha=0.35, label="Drawdown")
        ax.plot(dd.index, dd.values, color=_PDF_RED, linewidth=1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title("Drawdown — Out-of-Sample", color=_PDF_TEXT, fontsize=13, pad=12)
        ax.set_ylabel("Drawdown", color=_PDF_SUBTEXT)
        ax.tick_params(axis="x", labelcolor=_PDF_SUBTEXT, rotation=30)
        ax.tick_params(axis="y", labelcolor=_PDF_SUBTEXT)
        ax.grid(color="#21262d", linewidth=0.5)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 5 — Portfolio Weights Over Time (stacked area)
        # ------------------------------------------------------------------
        asset_cols = [c for c in wts.columns]
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(_PDF_BG)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=_PDF_SUBTEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        weight_data = [wts[c].values for c in asset_cols]
        colors_used = [_PDF_COLORS[i % len(_PDF_COLORS)] for i in range(len(asset_cols))]
        ax.stackplot(wts.index, weight_data, labels=asset_cols, colors=colors_used, alpha=0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(0, 1)
        ax.set_title("Portfolio Weights Over Time", color=_PDF_TEXT, fontsize=13, pad=12)
        ax.set_ylabel("Weight", color=_PDF_SUBTEXT)
        ax.tick_params(axis="x", labelcolor=_PDF_SUBTEXT, rotation=30)
        ax.tick_params(axis="y", labelcolor=_PDF_SUBTEXT)
        ax.legend(facecolor="#21262d", edgecolor="#30363d",
                  labelcolor=_PDF_TEXT, fontsize=9,
                  loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(color="#21262d", linewidth=0.5, axis="y")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 6 — Portfolio Manager Interpretation
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor(_PDF_BG)
        ax.set_facecolor(_PDF_BG)
        ax.axis("off")

        ax.text(0.5, 0.96, "Portfolio Manager Interpretation",
                ha="center", va="top", fontsize=16, fontweight="bold",
                color=_PDF_TEXT, transform=ax.transAxes)
        ax.text(0.5, 0.925, strategy,
                ha="center", va="top", fontsize=11, color=_PDF_SUBTEXT,
                transform=ax.transAxes)

        lines = []

        # Flag 1: Turnover
        rebal_events = folds_meta[folds_meta["rebalanced"]]
        span_yrs = 1.0  # safe default; overwritten below if rebalancing occurred
        ann_to = 0.0
        if not rebal_events.empty:
            mean_to = rebal_events["turnover"].mean() * 100
            span_yrs = max(
                ((rebal_events["test_date"].max() -
                  rebal_events["test_date"].min()).days + 30) / 365.25,
                0.5,
            )
            ann_to = (len(rebal_events) / span_yrs) * mean_to
            if ann_to > 200:
                lines.append(("⚡ HIGH TURNOVER", _PDF_ORANGE,
                               f"Annual turnover ≈ {ann_to:.0f}%. Consider reducing "
                               "rebalancing frequency or relaxing constraints."))
            else:
                lines.append(("✓ TURNOVER OK", _PDF_GREEN,
                               f"Annual turnover ≈ {ann_to:.0f}% — within normal range."))

        # Flag 2: Concentration
        last_w_pdf = wts.iloc[-1]
        max_w_pdf  = float(last_w_pdf.max())
        max_a_pdf  = str(last_w_pdf.idxmax())
        if max_w_pdf > 0.40:
            lines.append(("⚠ HIGH CONCENTRATION", _PDF_RED,
                           f"{max_a_pdf} holds {max_w_pdf:.1%} of the portfolio. "
                           "Consider lowering the max weight constraint."))
        else:
            lines.append(("✓ DIVERSIFIED", _PDF_GREEN,
                           f"Max weight: {max_a_pdf} at {max_w_pdf:.1%} — well diversified."))

        # Flag 3: Sharpe quality
        sharpe_val = summary.get("sharpe", float("nan"))
        if not np.isnan(sharpe_val):
            if sharpe_val > 1.0:
                lines.append(("✓ STRONG SHARPE", _PDF_GREEN,
                               f"Sharpe ratio {sharpe_val:.2f} exceeds 1.0 — strong risk-adjusted performance."))
            elif sharpe_val > 0:
                lines.append(("~ MODERATE SHARPE", _PDF_ORANGE,
                               f"Sharpe ratio {sharpe_val:.2f} — positive but below 1.0. "
                               "Review factor exposure and cost drag."))
            else:
                lines.append(("✗ NEGATIVE SHARPE", _PDF_RED,
                               f"Sharpe ratio {sharpe_val:.2f} — strategy destroys risk-adjusted value."))

        # Flag 4: Max drawdown
        mdd_val = summary.get("max_drawdown", float("nan"))
        if not np.isnan(mdd_val):
            if mdd_val < -0.30:
                lines.append(("⚠ LARGE DRAWDOWN", _PDF_RED,
                               f"Max drawdown {mdd_val:.1%}. Consider volatility targeting "
                               "or tighter stop-loss rules."))
            else:
                lines.append(("✓ DRAWDOWN MANAGEABLE", _PDF_GREEN,
                               f"Max drawdown {mdd_val:.1%} — within acceptable range."))

        # Recommendations
        lines.append(("", _PDF_SUBTEXT, ""))
        lines.append(("RECOMMENDATIONS", _PDF_BLUE, ""))
        recs = [
            f"1. Rebalancing freq: every {int(config.get('rebalance_months', 1))} month(s) — "
            + ("review if turnover is high." if ann_to > 200
               else "frequency appears appropriate."),
            "2. Evaluate adding risk-budget constraints to control tail risk.",
            "3. Back-test against the benchmark across multiple sub-periods.",
            "4. Monitor weight drift between rebalances, especially in volatile markets.",
            "5. Consider sensitivity analysis across δ and τ parameters (BL only).",
        ]
        for r in recs:
            lines.append(("", _PDF_TEXT, r))

        y_pos = 0.88
        for flag, color, body in lines:
            if flag == "" and body == "":
                y_pos -= 0.018
                continue
            if flag:
                ax.text(0.07, y_pos, flag,
                        ha="left", va="top", fontsize=9, fontweight="bold",
                        color=color, transform=ax.transAxes)
            if body:
                x_off = 0.07 if not flag else 0.30
                wrap_chars = 75 if not flag else 65
                words = body.split()
                current = ""
                wrapped = []
                for word in words:
                    if len(current) + len(word) + 1 <= wrap_chars:
                        current = (current + " " + word).strip()
                    else:
                        if current:
                            wrapped.append(current)
                        current = word
                if current:
                    wrapped.append(current)
                for line_txt in wrapped:
                    ax.text(x_off, y_pos, line_txt,
                            ha="left", va="top", fontsize=9,
                            color=_PDF_TEXT, transform=ax.transAxes)
                    y_pos -= 0.032
            else:
                y_pos -= 0.042

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Main — on Run
# ---------------------------------------------------------------------------
if run:
    # Load prices
    if data_source == "📂 Upload CSV":
        if uploaded_file is None:
            st.error("Please upload a CSV file.")
            st.stop()
        prices = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        prices.columns = [c.strip() for c in prices.columns]
    else:
        if not selected_tickers:
            st.error("Please enter at least one ticker.")
            st.stop()
        with st.spinner("Downloading prices…"):
            prices = download_prices(selected_tickers, str(start_date), str(end_date))
        if prices.empty:
            st.error("No data. Check tickers and date range.")
            st.stop()

    if prices.isnull().values.any():
        st.warning("Missing values detected — forward-filling.")
        prices = prices.ffill()

    prices_m = resample_to_month_end(prices)
    returns  = prices_to_log_returns(prices_m).dropna()

    if len(returns) < min_train_months + 2:
        st.error(f"Not enough data. Need {min_train_months + 2}+ months.")
        st.stop()

    # ── Build BL strategy if selected ─────────────────────────────────
    actual_strategy_name = strategy
    if is_bl:
        bl_views = st.session_state.get("bl_views", [])
        w_mkt = None
        if use_custom_mkt and custom_mkt_weights:
            w_mkt = pd.Series(custom_mkt_weights)

        view_confidences = [
            float(v.get("confidence", 0.5)) for v in bl_views
        ]
        bl_fn = make_bl_strategy(
            views=bl_views,
            w_mkt=w_mkt,
            delta=bl_delta,
            tau=bl_tau,
            min_weight=bl_min_w,
            max_weight=bl_max_w,
            view_confidences=view_confidences,
        )
        # Temporarily replace the BL strategy in the registry
        # (the factory wraps **kwargs silently)
        STRATEGIES["Black-Litterman"] = lambda rw, **kw: bl_fn(rw)

    # ── Effective weight bounds ────────────────────────────────────────
    eff_min_w = wc_min
    eff_max_w = wc_max

    # Validate feasibility
    n_assets = len(returns.columns)
    if eff_min_w * n_assets > 1.0:
        st.error(
            f"⚠️ Weight constraint infeasible: min_weight ({eff_min_w}) × "
            f"n_assets ({n_assets}) = {eff_min_w * n_assets:.2f} > 1.0"
        )
        st.stop()

    # Run primary strategy
    with st.spinner(f"Running {strategy} — {wf_method} window…"):
        results, folds_meta = run_walk_forward(
            returns,
            strategy=strategy,
            min_train_months=int(min_train_months),
            method=wf_method,
            transaction_cost_bps=float(transaction_cost_bps),
            rebalance_months=int(rebalance_months),
            min_weight=eff_min_w,
            max_weight=eff_max_w,
            rebalancing_type=rebalancing_type,
            drift_threshold=float(drift_threshold),
        )

    if results.empty:
        st.error("No results. Try a longer date range.")
        st.stop()

    strat  = results["Strategy"]
    wealth = np.exp(strat.cumsum())
    dd     = (wealth - wealth.cummax()) / wealth.cummax()
    wts    = results.drop(columns=["Strategy"])
    n_oos  = len(strat)

    if n_oos < 36:
        st.warning(f"⚠️ Only **{n_oos} OOS months** — below the 36-month reliability threshold.")

    # Benchmark
    _bench_start = start_date if start_date else prices.index[0].date()
    _bench_end   = end_date   if end_date   else prices.index[-1].date()
    bench_returns = _load_benchmark(benchmark_ticker, _bench_start, _bench_end)

    # Risk-free rate: convert annual % → monthly log-return series
    rf_series = None
    if rf_annual_pct > 0:
        rf_monthly = np.log(1 + rf_annual_pct / 100) / 12
        rf_series = pd.Series(rf_monthly, index=strat.index)

    # Metrics
    summary = compute_summary(strat, bench_returns, rf=rf_series)

    # Comparison strategies
    comparison_wealth: dict[str, pd.Series] = {}
    if compare_strategies:
        with st.spinner("Running comparison strategies…"):
            for cs in compare_strategies:
                cr, _ = run_walk_forward(
                    returns, strategy=cs,
                    min_train_months=int(min_train_months), method=wf_method,
                    transaction_cost_bps=float(transaction_cost_bps),
                    rebalance_months=int(rebalance_months),
                    min_weight=eff_min_w,
                    max_weight=eff_max_w,
                    rebalancing_type=rebalancing_type,
                    drift_threshold=float(drift_threshold),
                )
                if not cr.empty:
                    comparison_wealth[cs] = np.exp(cr["Strategy"].cumsum())

    # Rebalancing dates
    rebal_dates = folds_meta.loc[folds_meta["rebalanced"], "test_date"]
    wts_rebal   = wts.loc[wts.index.isin(rebal_dates)]

    # ====================================================================
    # Info banner
    # ====================================================================
    st.markdown(
        f"**{strategy}** &nbsp;·&nbsp; {wf_method.capitalize()} window "
        f"&nbsp;·&nbsp; {transaction_cost_bps} bps "
        f"&nbsp;·&nbsp; Rebalancing: `{rebalancing_type}`"
        + (f" (drift ≥ {drift_threshold:.0%})" if rebalancing_type == "drift_band" else "")
        + f" &nbsp;·&nbsp; {results.index[0].strftime('%b %Y')} → {results.index[-1].strftime('%b %Y')} "
        f"&nbsp;·&nbsp; **{n_oos} OOS months**"
    )
    st.divider()

    # ====================================================================
    # TABS — dynamic based on strategy
    # ====================================================================
    if is_bl:
        tab1, tab2, tab3 = st.tabs([
            "📈  Performance",
            "📊  Metrics & Methodology",
            "🔬  Black-Litterman Analysis",
        ])
    else:
        tab1, tab2 = st.tabs(["📈  Performance", "📊  Metrics & Methodology"])
        tab3 = None

    # ------------------------------------------------------------------
    # TAB 1 — Performance  (2×2 grid layout)
    # ------------------------------------------------------------------
    with tab1:

        # ── ROW 0: Mini KPI strip ───────────────────────────────────
        _k1, _k2, _k3, _k4, _k5 = st.columns(5)
        _k1.metric("Annualised Return",  f"{summary['annualized_return']:.2%}")
        _k2.metric("Sharpe",             f"{summary['sharpe']:.3f}")
        _k3.metric("Max Drawdown",       f"{summary['max_drawdown']:.2%}")
        _k4.metric("Sortino",            f"{summary['sortino']:.3f}")
        _k5.metric("Total Return",       f"{summary['total_return']:.2%}")
        st.divider()

        # ── ROW 1: Cumulative Wealth  |  Drawdown ────────────────
        _col_a, _col_b = st.columns(2, gap="medium")
        with _col_a:
            st.markdown("**Cumulative Wealth**")
            st.plotly_chart(_wealth_fig(wealth, strategy, comparison_wealth),
                            width="stretch")
        with _col_b:
            st.markdown("**Drawdown**")
            st.plotly_chart(_drawdown_fig(dd), width="stretch")

        # ── ROW 2: Portfolio Weights Over Time  |  Gross vs Net ─────
        _col_c, _col_d = st.columns(2, gap="medium")
        with _col_c:
            st.markdown("**Portfolio Weights Over Time**")
            st.caption("Stacked area with drift between rebalances.")
            st.plotly_chart(_stacked_area_fig(wts), width="stretch")
        with _col_d:
            st.markdown("**Gross vs Net Performance**")
            st.caption("Shaded area = cost drag from transaction costs.")
            with st.spinner("Computing gross returns…"):
                g_wealth = _gross_vs_net_fig(
                    returns, strategy,
                    int(min_train_months), wf_method,
                    int(rebalance_months),
                    eff_min_w, eff_max_w,
                    rebalancing_type=rebalancing_type,
                    drift_threshold=float(drift_threshold),
                )
            if g_wealth is not None and not g_wealth.empty:
                gn_fig = go.Figure()
                # Draw Net FIRST (below), then Gross on top with fill
                # so the green Gross line stays visible.
                gn_fig.add_trace(go.Scatter(
                    x=wealth.index, y=wealth.values,
                    mode="lines", name="Net (after costs)",
                    line=dict(color="#58a6ff", width=2),
                ))
                gn_fig.add_trace(go.Scatter(
                    x=g_wealth.index, y=g_wealth.values,
                    mode="lines", name="Gross (0 bps)",
                    line=dict(color="#3fb950", width=2.5),
                    fill="tonexty",
                    fillcolor="rgba(247,129,102,0.15)",
                ))
                gn_fig.update_layout(
                    **DARK_LAYOUT, height=340,
                    yaxis_title="Wealth", hovermode="x unified",
                    legend=dict(orientation="h", y=1.08, font=dict(color="#e6edf3", size=11)),
                )
                st.plotly_chart(gn_fig, width="stretch")
            else:
                st.info("Could not compute gross returns.")

        # ── ROW 3: Turnover  |  Risk Contribution ────────────────
        _col_e, _col_f = st.columns(2, gap="medium")
        with _col_e:
            st.markdown("**Turnover per Rebalancing Date**")
            to_result = _turnover_fig(folds_meta)
            if to_result:
                to_fig, annual_to = to_result
                st.plotly_chart(to_fig, width="stretch")
            else:
                st.info("No rebalancing events.")

        with _col_f:
            st.markdown("**Risk Contribution per Asset**")
            st.caption("Marginal Risk Contribution from last rebalancing weights.")
            try:
                _last_w  = wts.iloc[-1]
                _cov_m   = returns.cov()
                _w_arr   = _last_w.values.astype(float)
                _cov_arr = _cov_m.values.astype(float)
                _port_var = float(_w_arr @ _cov_arr @ _w_arr)
                if _port_var > 1e-12:
                    _port_vol    = float(np.sqrt(_port_var))
                    _mrc_raw     = _w_arr * (_cov_arr @ _w_arr) / _port_vol
                    _mrc_sum     = _mrc_raw.sum()
                    _risk_budget = pd.Series(
                        _mrc_raw / _mrc_sum if _mrc_sum > 1e-12 else _mrc_raw,
                        index=_last_w.index,
                    )
                    _rb_assets = _risk_budget.index.tolist()
                    _rb_fig = go.Figure(go.Bar(
                        x=(_risk_budget.values * 100).tolist(),
                        y=_rb_assets,
                        orientation="h",
                        marker_color=[COLORS[i % len(COLORS)] for i in range(len(_rb_assets))],
                        text=[f"{v:.1f}%" for v in _risk_budget.values * 100],
                        textposition="outside",
                        textfont=dict(color="#e6edf3"),
                        hovertemplate="%{x:.2f}%<extra></extra>",
                    ))
                    _equal_share = 100.0 / len(_rb_assets)
                    _rb_fig.add_vline(
                        x=_equal_share,
                        line_dash="dash", line_color="#ffa657",
                        annotation_text=f"Equal ({_equal_share:.1f}%)",
                        annotation_font_color="#ffa657",
                        annotation_font_size=10,
                    )
                    _rb_fig.update_layout(
                        **DARK_LAYOUT,
                        height=max(220, len(_rb_assets) * 52),
                        xaxis_title="Risk Contribution (%)",
                        xaxis_ticksuffix="%",
                    )
                    st.plotly_chart(_rb_fig, width="stretch")

                    # Interpretation badge
                    _max_rc      = float(_risk_budget.max())
                    _max_rc_asset = _risk_budget.idxmax()
                    _max_rc_weight = float(_last_w[_max_rc_asset])
                    if _max_rc > 0.50:
                        st.warning(
                            f"⚠️ **{_max_rc_asset}** contributes **{_max_rc:.0%}** of risk "
                            f"(weight: {_max_rc_weight:.1%}). Consider a risk-budget constraint."
                        )
                    else:
                        st.success("✅ Risk is reasonably distributed across assets.")
                else:
                    st.info("Cannot compute risk contributions (near-zero variance).")
            except Exception as _rb_err:
                st.info(f"Risk contribution unavailable: {_rb_err}")

        st.divider()

        # ── ROW 4: Portfolio Weights at Rebalancing Dates (full width) ──
        st.markdown("**Portfolio Weights at Rebalancing Dates**")
        if not wts_rebal.empty:
            st.plotly_chart(_weights_fig(wts_rebal, strategy), width="stretch")
        else:
            st.info("No rebalancing events to display.")

        st.divider()

        # ── ROW 5: Rebalancing Mode Comparison (collapsible) ─────────
        with st.expander("⚖️ Rebalancing Mode Comparison", expanded=False):
            st.caption(
                "Runs all 4 modes on the same data. "
                "Target weights w∗ are recomputed every step; only trading timing differs."
            )

            _rebal_modes = ["monthly", "drift_band", "yearly"]

            _mode_results: dict[str, pd.DataFrame] = {}
            with st.spinner("Running rebalancing mode comparison…"):
                for _mode in _rebal_modes:
                    # Only drift_band uses drift_threshold; others get 0.0
                    _dt = float(drift_threshold) if _mode == "drift_band" else 0.0
                    _kw = dict(
                        returns=returns,
                        strategy=strategy,
                        min_train_months=int(min_train_months),
                        method=wf_method,
                        transaction_cost_bps=float(transaction_cost_bps),
                        rebalance_months=int(rebalance_months),
                        min_weight=eff_min_w,
                        max_weight=eff_max_w,
                        rebalancing_type=_mode,
                        drift_threshold=_dt,
                    )
                    try:
                        _mr, _mf = run_walk_forward(**_kw)
                        if not _mr.empty:
                            _mode_results[_mode] = {"results": _mr, "folds": _mf}
                    except Exception:
                        pass

            if _mode_results:
                # ---- Metrics table ----------------------------------------
                _comparison_rows = []
                for _mode, _data in _mode_results.items():
                    _mr   = _data["results"]
                    _mf   = _data["folds"]
                    _rets = _mr["Strategy"]

                    _sharpe = (
                        float(_rets.mean() / _rets.std() * np.sqrt(12))
                        if _rets.std() > 0 else float("nan")
                    )
                    _wealth_m = np.exp(_rets.cumsum())
                    _peaks_m  = _wealth_m.cummax()
                    _mdd = float(((_wealth_m - _peaks_m) / _peaks_m).min())

                    _rebal_m = _mf[_mf["rebalanced"]]
                    _ann_to = 0.0
                    if not _rebal_m.empty:
                        _mean_to = _rebal_m["turnover"].mean() * 100
                        _span_y  = max(
                            ((_rebal_m["test_date"].max() -
                              _rebal_m["test_date"].min()).days + 30) / 365.25, 0.5,
                        )
                        _ann_to = (len(_rebal_m) / _span_y) * _mean_to

                    _cum_cost = float(_mf["cost_bps"].sum() / 100)

                    _comparison_rows.append({
                        "Mode":             _mode,
                        "Sharpe":           round(_sharpe, 3),
                        "Max DD":           _mdd,
                        "Ann. Turnover (%)": round(_ann_to, 1),
                        "Cum. Cost (%)": round(_cum_cost, 2),
                    })

                _comp_df = pd.DataFrame(_comparison_rows).set_index("Mode")

                def _highlight_rebal_table(df):
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    if "Sharpe" in df.columns and df["Sharpe"].notna().any():
                        best = df["Sharpe"].idxmax()
                        styles.loc[best, "Sharpe"] = "background-color:rgba(63,185,80,0.25);color:#3fb950;font-weight:700"
                    if "Max DD" in df.columns:
                        least_bad = df["Max DD"].idxmax()
                        styles.loc[least_bad, "Max DD"] = "background-color:rgba(63,185,80,0.25);color:#3fb950;font-weight:700"
                    if "Ann. Turnover (%)" in df.columns:
                        lowest = df["Ann. Turnover (%)"].idxmin()
                        styles.loc[lowest, "Ann. Turnover (%)"] = "background-color:rgba(88,166,255,0.20);color:#58a6ff;font-weight:700"
                    return styles

                _styled_comp = _comp_df.style.apply(
                    _highlight_rebal_table, axis=None
                ).format({
                    "Sharpe":            "{:.3f}",
                    "Max DD":            "{:.1%}",
                    "Ann. Turnover (%)": "{:.1f}%",
                    "Cum. Cost (%)": "{:.2f}%",
                })

                # Side-by-side: table left, overlay chart right
                _rc1, _rc2 = st.columns([1, 1], gap="medium")
                with _rc1:
                    st.dataframe(_styled_comp, width="stretch")
                    if _comparison_rows:
                        _best_s = max(_comparison_rows, key=lambda r: r["Sharpe"] if not np.isnan(r["Sharpe"]) else -999)["Mode"]
                        _low_t  = min(_comparison_rows, key=lambda r: r["Ann. Turnover (%)"])["Mode"]
                        st.info(f"💡 Best Sharpe: **{_best_s}** — Lowest turnover: **{_low_t}**")

                with _rc2:
                    st.markdown("**Cumulative Wealth by Mode**")
                    _ov_fig = go.Figure()
                    _mode_colors = {"monthly": "#58a6ff", "drift_band": "#ffa657", "yearly": "#d2a8ff"}
                    _mode_dash  = {"monthly": "solid", "drift_band": "dash", "yearly": "dashdot"}
                    for _mode, _data in _mode_results.items():
                        _w = np.exp(_data["results"]["Strategy"].cumsum())
                        _label = f"drift_band ({drift_threshold:.0%})" if _mode == "drift_band" else _mode
                        _ov_fig.add_trace(go.Scatter(
                            x=_w.index, y=_w.values, mode="lines", name=_label,
                            line=dict(color=_mode_colors.get(_mode, "#fff"), width=2.2, dash=_mode_dash.get(_mode, "solid")),
                            hovertemplate="%{y:.3f}<extra>" + _label + "</extra>",
                        ))
                    _ov_fig.update_layout(
                        **DARK_LAYOUT, height=320,
                        yaxis_title="Wealth", hovermode="x unified",
                        legend=dict(orientation="h", y=1.1, font=dict(color="#e6edf3")),
                    )
                    st.plotly_chart(_ov_fig, width="stretch")

    # ------------------------------------------------------------------
    # TAB 2 — Metrics & Methodology
    # ------------------------------------------------------------------
    with tab2:
        col_metrics, col_method = st.columns([1, 1], gap="large")

        # ---- Left: Metrics ----
        with col_metrics:
            st.markdown("### Performance Metrics")

            def _fmt(val, pct=False):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                return f"{val:.2%}" if pct else f"{val:.3f}"

            bench_label = benchmark_ticker.strip() or "Benchmark"
            metric_rows = [
                ("Annualized Return",                  _fmt(summary["annualized_return"], pct=True)),
                ("Sharpe Ratio",                       _fmt(summary["sharpe"])),
                ("Sortino Ratio",                      _fmt(summary["sortino"])),
                ("Max Drawdown",                       _fmt(summary["max_drawdown"], pct=True)),
                ("Calmar Ratio",                       _fmt(summary["calmar"])),
                (f"Beta vs {bench_label}",             _fmt(summary["beta"])),
                ("Information Ratio",                  _fmt(summary["information_ratio"])),
                ("Total Return",                       _fmt(summary["total_return"], pct=True)),
            ]
            # Display in 2-column grid
            for i in range(0, len(metric_rows), 2):
                c1, c2 = st.columns(2)
                c1.metric(metric_rows[i][0], metric_rows[i][1])
                if i + 1 < len(metric_rows):
                    c2.metric(metric_rows[i+1][0], metric_rows[i+1][1])

            st.divider()
            st.markdown("### Latest Weights")
            last_w = wts.iloc[-1].sort_values(ascending=False)
            st.dataframe(
                last_w.rename("Allocation").to_frame().style.format("{:.1%}"),
                width="stretch"
            )

            st.divider()
            st.markdown("### Backtest Stats")
            st.markdown(f"""
| | |
|---|---|
| OOS months | **{n_oos}** |
| Rebalancing events | **{int(folds_meta['rebalanced'].sum())}** |
| Avg turnover / rebal | **{folds_meta.loc[folds_meta['rebalanced'], 'turnover'].mean():.1%}** |
| Avg cost / rebal | **{folds_meta.loc[folds_meta['rebalanced'], 'cost_bps'].mean():.1f} bps** |
""")

            st.divider()
            st.markdown("### Download")
            st.download_button("⬇ Backtest Results (CSV)",
                               results.to_csv().encode(), f"backtest_{strategy}.csv", "text/csv",
                               width="stretch")
            st.download_button("⬇ Walk-Forward Folds (CSV)",
                               folds_meta.to_csv(index=False).encode(), "folds.csv", "text/csv",
                               width="stretch")
            st.download_button("⬇ Prices (CSV)",
                               prices.to_csv().encode(), "prices.csv", "text/csv",
                               width="stretch")

        # ---- Right: Methodology ----
        with col_method:
            st.markdown("### Walk-Forward Methodology")
            st.markdown(
                "Each month, the model is trained on **all available history up to t-1**, "
                "then tested on **month t** — which it has never seen."
            )
            st.code(
                f"Fold 1: Train [t0 → t{min_train_months-1}]  →  Test [t{min_train_months}]\n"
                f"Fold 2: Train [t0 → t{min_train_months}]  →  Test [t{min_train_months+1}]\n"
                "  ...\n"
                "Fold N: Train [t0 → tN-1]  →  Test [tN]",
                language=None,
            )
            st.markdown(
                "✅ Every return is **Out-of-Sample**  \n"
                "✅ Training window **grows** at each step  \n"
                "✅ Transaction costs applied on **turnover**  \n"
                "✅ Weights **drift** with market returns between rebalances"
            )

            st.markdown("### Walk-Forward Timeline")
            gantt_fig, gantt_step = _gantt_fig(folds_meta)
            st.plotly_chart(gantt_fig, width="stretch")
            st.caption(
                f"🔵 Blue = training period · 🟠 Orange = OOS test month. "
                f"Showing 1 in every {gantt_step} fold(s) — {len(folds_meta)} total."
            )

            st.divider()
            st.markdown("### Download Report")
            with st.spinner("Generating PDF…"):
                try:
                    _pdf_bytes = _generate_pdf_report(
                        strategy=strategy,
                        summary=summary,
                        wts=wts,
                        strat=strat,
                        folds_meta=folds_meta,
                        wealth=wealth,
                        dd=dd,
                        comparison_wealth=comparison_wealth,
                        config={
                            "tickers":               selected_tickers,
                            "start":                 str(results.index[0].date()),
                            "end":                   str(results.index[-1].date()),
                            "transaction_cost_bps":  transaction_cost_bps,
                            "wf_method":             wf_method,
                            "min_weight":            eff_min_w,
                            "max_weight":            eff_max_w,
                            "rebalance_months":      int(rebalance_months),
                        },
                    )
                    _pdf_filename = (
                        f"strategy_lab_"
                        f"{strategy.lower().replace(' ', '_').replace('-', '_')}"
                        f"_report.pdf"
                    )
                    st.download_button(
                        label="📄 Download Report (PDF)",
                        data=_pdf_bytes,
                        file_name=_pdf_filename,
                        mime="application/pdf",
                        width="stretch",
                    )
                except Exception as _pdf_err:
                    st.error(f"PDF generation failed: {_pdf_err}")

    # ------------------------------------------------------------------
    # TAB 3 — Black-Litterman Analysis (only for BL strategy)
    # ------------------------------------------------------------------
    if is_bl and tab3 is not None:
        with tab3:
            bl_views = st.session_state.get("bl_views", [])
            assets = list(returns.columns)
            n_a = len(assets)

            # Build data from the last training window
            last_window = returns.iloc[:-1]  # all history except last month
            cov = last_window.cov().values * 12  # annualize monthly covariance (matches engine pipeline)

            # Market weights
            w_mkt_arr = np.ones(n_a) / n_a
            if use_custom_mkt and custom_mkt_weights:
                w_s = pd.Series(custom_mkt_weights).reindex(assets).fillna(0.0)
                total = w_s.sum()
                if total > 1e-12:
                    w_mkt_arr = (w_s / total).values

            # Implied equilibrium returns
            from strategy_lab.black_litterman import (
                implied_equilibrium_returns, build_view_matrices, bl_posterior,
            )
            pi = implied_equilibrium_returns(cov, w_mkt_arr, delta=bl_delta)

            # BL posterior
            has_views = len(bl_views) > 0
            if has_views:
                try:
                    P, Q, Omega = build_view_matrices(bl_views, assets, cov, tau=bl_tau)
                    mu_bl, _ = bl_posterior(pi, cov, P, Q, Omega, tau=bl_tau)
                except Exception:
                    mu_bl = pi.copy()
                    has_views = False
            else:
                mu_bl = pi.copy()

            # ── BL charts in 2-column layout ─────────────────────────
            _bl_left, _bl_right = st.columns(2, gap="medium")

            with _bl_left:
                st.markdown("**Implied Equilibrium vs BL Posterior Returns**")
                if has_views:
                    st.caption("Green bars pulled toward your views; blue = equilibrium prior.")
                else:
                    st.caption("No views — BL posterior equals the equilibrium prior.")
                st.plotly_chart(_bl_returns_fig(assets, pi, mu_bl), width="stretch")

            with _bl_right:
                st.markdown("**Markowitz vs Black-Litterman Weights**")
                st.caption("Unconstrained Markowitz (orange) vs BL allocation (green).")
                comp_df = compare_bl_vs_markowitz(
                    last_window, bl_views,
                    w_mkt=pd.Series(custom_mkt_weights) if use_custom_mkt and custom_mkt_weights else None,
                    delta=bl_delta, tau=bl_tau,
                    min_weight=bl_min_w, max_weight=bl_max_w,
                ).set_index("asset")
                st.plotly_chart(_bl_weights_comparison_fig(comp_df), width="stretch")


            # Detail table
            with st.expander("📋 Detailed comparison table"):
                # Color-highlight BL return > implied
                styled = comp_df.style.format({
                    "markowitz_weight": "{:.2%}",
                    "bl_weight": "{:.2%}",
                    "implied_return": "{:.4%}",
                    "bl_return": "{:.4%}",
                })
                st.dataframe(styled, width="stretch")

            # ── Section C: Sensitivity Analysis δ × τ Grid ───────────
            st.markdown("### Sensitivity Analysis — δ × τ Grid")
            st.caption(
                "Sharpe ratio across combinations of risk aversion (δ) "
                "and prior uncertainty (τ). Brighter = better Sharpe."
            )

            if "sens_df" not in st.session_state:
                st.session_state["sens_df"] = None

            if st.button("▶ Run sensitivity analysis (δ × τ grid)", key="run_sensitivity"):
                with st.spinner("Running 24-combination grid…"):
                    try:
                        _sens_df = run_sensitivity_analysis(
                            returns=returns,
                            views=bl_views,
                            w_mkt=w_mkt,
                            delta_values=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                            tau_values=[0.01, 0.05, 0.10, 0.20],
                            view_confidences=(
                                [v.get("confidence", 0.5) for v in bl_views]
                                if bl_views else [0.5]
                            ),
                            min_weight=bl_min_w,
                            max_weight=bl_max_w,
                        )
                        st.session_state["sens_df"] = _sens_df
                    except Exception as _sens_ex:
                        st.error(f"Sensitivity analysis failed: {_sens_ex}")

            _stored_sens = st.session_state.get("sens_df")
            if _stored_sens is not None and not _stored_sens.empty and "sharpe" in _stored_sens.columns:
                _pivot = _stored_sens["sharpe"].unstack(level="tau")
                _flat  = _pivot.values[~np.isnan(_pivot.values)]
                _mid   = float(_flat.mean()) if len(_flat) > 0 else 0.0
                _sens_fig = px.imshow(
                    _pivot,
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=_mid,
                    text_auto=".2f",
                    labels={
                        "x": "τ (prior uncertainty)",
                        "y": "δ (risk aversion)",
                        "color": "Sharpe",
                    },
                    aspect="auto",
                )
                _sens_fig.update_layout(
                    **DARK_LAYOUT, height=340,
                    xaxis=dict(tickfont=dict(color="#e6edf3", size=11)),
                    yaxis=dict(tickfont=dict(color="#e6edf3", size=11)),
                    coloraxis_colorbar=dict(
                        tickfont=dict(color="#adbac7", size=10),
                        title_font=dict(color="#adbac7"),
                    ),
                )
                st.plotly_chart(_sens_fig, width="stretch")

                _best_idx   = _stored_sens["sharpe"].idxmax()
                _best_sharpe = float(_stored_sens.loc[_best_idx, "sharpe"])
                st.success(
                    f"✅ Best Sharpe: δ={_best_idx[0]}, τ={_best_idx[1]} "
                    f"→ Sharpe = {_best_sharpe:.3f}"
                )

            # ── Section E: Portfolio Manager Notes ────────────────────
            st.markdown("---")
            with st.expander("🧠 Portfolio Manager Notes", expanded=True):

                # Flag 1: Excessive turnover
                rebal_events = folds_meta[folds_meta["rebalanced"]]
                if not rebal_events.empty:
                    mean_to_pct = rebal_events["turnover"].mean() * 100
                    span_yrs = ((rebal_events["test_date"].max() -
                                 rebal_events["test_date"].min()).days + 30) / 365.25
                    n_rebal = len(rebal_events)
                    annual_turnover = (n_rebal / max(span_yrs, 0.5)) * mean_to_pct
                    if annual_turnover > 200:
                        st.warning(
                            f"⚡ **High annual turnover**: ≈{annual_turnover:.0f}%. "
                            "Consider reducing rebalancing frequency or relaxing weight constraints."
                        )
                    else:
                        st.success(
                            f"✅ **Annual turnover** ≈{annual_turnover:.0f}% — within normal range."
                        )

                # Flag 2: View consistency
                if has_views:
                    deviations = np.abs(mu_bl - pi) / (np.abs(pi) + 1e-8)
                    max_dev_idx = int(np.argmax(deviations))
                    max_dev = deviations[max_dev_idx]
                    if max_dev > 0.50:
                        st.warning(
                            f"🔍 **Large view deviation**: {assets[max_dev_idx]} BL return "
                            f"deviates {max_dev:.0%} from equilibrium. "
                            "Consider reducing the view magnitude or increasing τ."
                        )
                    else:
                        st.success(
                            "✅ **Views are consistent** with implied equilibrium returns — "
                            f"max deviation is {max_dev:.0%}."
                        )
                else:
                    st.info("ℹ️ No active views — the portfolio follows the equilibrium prior.")

                # Flag 3: Concentration
                last_weights = wts.iloc[-1]
                max_w_val = last_weights.max()
                max_w_asset = last_weights.idxmax()
                if max_w_val > 0.40:
                    st.warning(
                        f"⚠️ **High concentration**: {max_w_asset} has {max_w_val:.1%} weight. "
                        "Consider lowering the max weight constraint."
                    )
                else:
                    st.success(
                        f"✅ **Well diversified**: max weight is {max_w_asset} at {max_w_val:.1%}."
                    )

else:
    # Landing state
    st.markdown("""
<div style="text-align:center; padding: 80px 20px; color: #8b949e;">
    <div style="font-size: 3rem; margin-bottom: 1rem;">💹</div>
    <h2 style="color: #e6edf3; font-weight: 600;">Ready to backtest</h2>
    <p>Configure your assets and strategy in the sidebar,<br>then click <strong>▶ Run backtest</strong>.</p>
    <br>
    <p style="font-size:0.85rem;">
        Walk-forward · Expanding window · 100% Out-of-Sample
    </p>
</div>
""", unsafe_allow_html=True)
